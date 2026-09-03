"""
lstm_v2.py — improved LSTM + self-attention next-item recommender.

Companion to (not a replacement for) ``src/lstm.py::LSTMAttentionModel``.
It applies the Tier-1 / Tier-2 changes from the baseline review plus the
Tier-3 repeat/recency mechanism, and nothing else from Tier-3/4 (no
GRU/SASRec swap, no FFN transformer block, no metadata fusion, no
ensembling / re-ranking).

Structural changes vs. ``LSTMAttentionModel``
---------------------------------------------
Tier 1 — correctness & training signal
  * ``nn.Embedding(..., padding_idx=pad)`` — pad rows stay a zero vector and
    receive no gradient.
  * **Boolean** causal mask, sized to the *actual* sequence length at run
    time.  The old mask was ``torch.triu(torch.ones(ctx, ctx), diagonal=1)``
    — a float 0/1 tensor passed as ``attn_mask``, which PyTorch *adds* to the
    scores, so it never masked the future and broke whenever a batch was not
    exactly ``context_length`` long.
  * ``key_padding_mask`` derived from real sequence lengths, so attention no
    longer pools padding positions.
  * ``forward`` returns **per-step** logits ``(B, L, V)`` so the training loop
    can supervise next-item prediction at every position (~L× more gradient
    per session) instead of only the last step.

Tier 2 — capacity / optimisation levers that must live in the model
  * **Weight tying** between the input embedding and the output decoder
    (via a ``hidden→emb`` projection when ``hidden_dim != emb_dim``).
  * Removed the redundant hand-rolled Q/K/V ``nn.Linear`` layers —
    ``nn.MultiheadAttention`` already does its own input projection.

Tier 3 (5) — repeat / recency bias
  * A learned scalar ``recency_gamma`` adds ``gamma * decay**(t-s)`` to the
    logit of every item seen at an earlier (or the current) context position
    ``s <= t``.  This is the cheap "next item ≈ a recently seen item" signal
    that first-order Markov exploits and the pure neural model was missing
    (it lost to Markov on MRR on every dataset).  ``gamma`` starts at 0 (no
    effect) and the model learns its sign/size — it can go negative when
    repeats are *unlikely* (e.g. purchase-prediction targets). Toggle with
    ``use_recency_bias``; tune emphasis with ``recency_decay`` (1.0 = every
    prior occurrence counts equally, <1 = weight recent ones more).

The remaining Tier-1/2 items are training-loop / config concerns and live in
``train_lstm_v2.py`` and the YAML configs:
  * per-step cross-entropy with ``ignore_index`` (loss over the shifted seq),
  * low ``label_smoothing`` (~0.0–0.05) and low ``weight_decay`` (~0.01–0.05),
  * linear-warmup + cosine LR decay (replacing CosineAnnealingWarmRestarts),
  * early stopping / checkpointing on validation NDCG@K, not val loss,
  * recommended: ``emb_dim == hidden_dim`` (e.g. 128/128 or 256/256),
    ``n_heads`` 2–4, ``context_length`` ~20.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAttentionRec(nn.Module):
    def __init__(
        self,
        items_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 1,
        n_heads: int = 2,
        drop_rate: float = 0.2,
        pad_token_id: int = 0,
        tie_weights: bool = True,
        use_recency_bias: bool = True,
        recency_decay: float = 0.9,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.tie_weights = tie_weights
        self.use_recency_bias = use_recency_bias
        self.recency_decay = float(recency_decay)
        self._max_len = max_len

        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(items_size, emb_dim, padding_idx=pad_token_id)
        self.emb_drop = nn.Dropout(drop_rate)
        # Small init: the embedding matrix doubles as the output projection when
        # ``tie_weights`` is set, so an N(0, 1) default would blow up the logits.
        # The LSTM input path compensates with a sqrt(emb_dim) scale (see forward).
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[pad_token_id].zero_()

        lstm_dropout = drop_rate if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Self-attention over the LSTM hidden states (MHA does its own q/k/v proj).
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=drop_rate,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_drop = nn.Dropout(drop_rate)

        # ---- tied output decoder --------------------------------------------
        if tie_weights:
            self.h2e = nn.Linear(hidden_dim, emb_dim, bias=False) if hidden_dim != emb_dim else None
            self.decoder_bias = nn.Parameter(torch.zeros(items_size))
        else:
            self.fc = nn.Linear(hidden_dim, items_size)

        # ---- repeat / recency bias (Tier 3.5) ------------------------------
        # Learned scalar, starts at 0 so it is a no-op at init.
        if use_recency_bias:
            self.recency_gamma = nn.Parameter(torch.zeros(()))

        # Cached boolean causal mask (True = position is NOT allowed to attend).
        self.register_buffer(
            "_causal",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    # ----------------------------------------------------------------------
    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len <= self._max_len:
            return self._causal[:seq_len, :seq_len].to(device)
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )

    def _decode(self, h: torch.Tensor) -> torch.Tensor:
        if not self.tie_weights:
            return self.fc(h)
        if self.h2e is not None:
            h = self.h2e(h)
        return F.linear(h, self.embedding.weight, self.decoder_bias)

    @staticmethod
    def _infer_lengths(x: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        return (x != pad_token_id).sum(dim=1)

    def _recency_bias(
        self, logits: torch.Tensor, x: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Add ``gamma * decay**(t - s)`` to ``logits[b, t, x[b, s]]`` for every
        earlier / current context position ``s <= t`` (``s`` a real token).

        Encodes "the next item is often one we have just seen" — the cheap
        repeat/recency signal first-order Markov gets for free.
        """
        bsz, seq_len, _ = logits.shape
        pos = torch.arange(seq_len, device=x.device)
        dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()      # (L, L) t - s
        weight = torch.tril(self.recency_decay ** dist)                # (L, L), s <= t only
        valid = (pos[None, :] < lengths[:, None]).float()             # (B, L) real key positions
        src = self.recency_gamma * weight[None] * valid[:, None, :]    # (B, L, L) value at [t, s]
        index = x[:, None, :].expand(bsz, seq_len, seq_len)           # (B, L, L) -> item id x[b, s]
        return logits.scatter_add(2, index, src)

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """
        Args:
            x:       (B, L) right-padded token-id tensor.
            lengths: (B,) count of real tokens per row. Inferred (assuming
                     right-padding) when ``None``.

        Returns:
            logits:  (B, L, V) next-item logits at every position.
            (hn, cn): final LSTM states.
        """
        bsz, seq_len = x.shape
        if lengths is None:
            lengths = self._infer_lengths(x, self.pad_token_id)
        lengths = lengths.clamp(min=1)

        # sqrt(emb_dim) scale so the tiny (std=0.02) embeddings still drive the
        # LSTM at a sane magnitude; the tied decoder keeps using the raw matrix.
        emb = self.emb_drop(self.embedding(x) * (self.emb_dim ** 0.5))  # (B, L, E)

        h0 = emb.new_zeros((self.n_layers, bsz, self.hidden_dim))
        c0 = emb.new_zeros((self.n_layers, bsz, self.hidden_dim))
        lstm_out, (hn, cn) = self.lstm(emb, (h0, c0))                # (B, L, H)

        key_padding_mask = (
            torch.arange(seq_len, device=x.device)[None, :] >= lengths[:, None]
        )                                                            # (B, L) True = pad
        attn_out, _ = self.attn(
            lstm_out,
            lstm_out,
            lstm_out,
            attn_mask=self._causal_mask(seq_len, x.device),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = self.norm(attn_out + lstm_out)
        h = self.out_drop(h)
        logits = self._decode(h)                                     # (B, L, V)
        if self.use_recency_bias:
            logits = self._recency_bias(logits, x, lengths)
        return logits, (hn, cn)

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def predict_last(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        mask_token_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """Logits at the last *real* position → ``(B, V)``, for top-k inference."""
        if lengths is None:
            lengths = self._infer_lengths(x, self.pad_token_id)
        lengths = lengths.clamp(min=1)

        logits, _ = self.forward(x, lengths)                         # (B, L, V)
        gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, logits.size(-1))
        last = logits.gather(1, gather_idx).squeeze(1)               # (B, V)

        last[:, self.pad_token_id] = float("-inf")
        if mask_token_ids:
            last[:, mask_token_ids] = float("-inf")
        return last
