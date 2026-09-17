from __future__ import annotations

import torch
import torch.nn as nn
import json
import torch.nn.functional as F

def _build_meta_embedding(
    emb_dim: int,
    tokenizer,
    emb_path: str,
    id_map_path: str,
) -> tuple[nn.Embedding, torch.Tensor]:
    """
    Load pretrained item metadata embeddings and build a bridge tensor that
    maps tokenizer indices → metadata embedding rows.

    The metadata embedding has an extra row 0 (zero vector, padding_idx) for
    tokenizer entries that have no corresponding metadata (special tokens,
    unknown items). All real item rows are shifted by +1.

    Returns:
        meta_emb    nn.Embedding [num_items+1, emb_dim]
        tok_to_meta LongTensor   [vocab_size]  — registered as a buffer
    """
    pretrained = torch.load(emb_path, map_location="cpu", weights_only=True)
    num_items, pretrained_dim = pretrained.shape

    assert pretrained_dim == emb_dim, (
        f"Metadata embedding dim ({pretrained_dim}) != model emb_dim ({emb_dim}). "
        f"Re-run build_item_embeddings.py with --embed_dim {emb_dim}."
    )

    # Row 0 = zero padding; rows 1..N = pretrained embeddings (1-based shift)
    weight = torch.cat([torch.zeros(1, emb_dim), pretrained], dim=0)
    meta_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
    meta_emb.weight = nn.Parameter(weight)

    # id_map: {"item_id_str": metadata_row_index (0-based), ...}
    with open(id_map_path) as f:
        id_map: dict[str, int] = json.load(f)

    vocab = tokenizer.get_vocab()   # {"token_str": tokenizer_index, ...}
    vocab_size = tokenizer.get_vocab_size()

    # Build bridge: tokenizer_index → metadata_row (1-based), 0 = no metadata
    tok_to_meta = torch.zeros(vocab_size, dtype=torch.long)
    for token_str, tok_idx in vocab.items():
        if token_str in id_map:
            tok_to_meta[tok_idx] = id_map[token_str] + 1   # +1 for the padding row offset

    return meta_emb, tok_to_meta


class LSTMModel(nn.Module):

  def __init__(self, embedded_dim, hidden_dim, layer_dim, items_size, drop_rate:float=0):
    super(LSTMModel, self).__init__()

    self.hidden_dim = hidden_dim
    self.layer_dim = layer_dim


    self.embedding = nn.Embedding(items_size, embedded_dim)

    self.lstm = nn.LSTM(embedded_dim, hidden_dim, num_layers=layer_dim, dropout=drop_rate, batch_first=True)
    self.fc = nn.Linear(hidden_dim, items_size)

  def forward(self, x, h0=None, c0=None):

    # Initialize hidden state and cell state if not provided
    if h0 is None or c0 is None:
      h0 = torch.zeros(self.layer_dim, x.size(
          0), self.hidden_dim).to(x.device)
      c0 = torch.zeros(self.layer_dim, x.size(
          0), self.hidden_dim).to(x.device)


    embedded = self.embedding(x)

    """
    out (intermediate_hidden_states): Tensor of shape (batch_size, seq_length, hidden_size) containing the output features (h_t) from the last layer of the LSTM, for each t.
    hn (final_hidden_state): Tensor of shape (num_layers * num_directions, batch_size, hidden_size) containing the final hidden state for each element in the sequence.
    cn (final_cell_state): Tensor of shape (num_layers * num_directions, batch_size, hidden_size) containing the final cell state for each element in the sequence.
    
    NOTE: out will contain the hidden states for all time steps, while hn and cn will contain the hidden and cell states for the last time step only.
          it will be used as input to attention mechanism, while hn and cn will be used to initialize the hidden and cell states for the next batch of sequences.

    """


    out, (hn, cn) = self.lstm(embedded, (h0, c0))
    output = self.fc(hn.squeeze(0))
    return output, hn, cn



class LSTMAttentionModel(nn.Module):

  def __init__(self, embedded_dim, hidden_dim, layer_dim, items_size, n_head = 4, drop_rate = 0.1, context_length = 10):

    super(LSTMAttentionModel, self).__init__()

    self.hidden_dim = hidden_dim
    self.layer_dim = layer_dim
    self.kdim = hidden_dim
    self.vdim = hidden_dim


    self.embedding = nn.Embedding(items_size, embedded_dim)

    if layer_dim > 1:
        self.lstm = nn.LSTM(embedded_dim, hidden_dim, num_layers=layer_dim, batch_first=True, dropout=drop_rate)
    else:
        self.lstm = nn.LSTM(embedded_dim, hidden_dim, num_layers=layer_dim, batch_first=True)

    # multihead attention
    self.q = nn.Linear(hidden_dim, self.kdim)
    self.k = nn.Linear(hidden_dim, self.kdim)
    self.v = nn.Linear(hidden_dim, self.vdim)

    # Correct: Registering the tensor as a buffer. This ensures that the tensor is moved to the appropriate device when the model is moved to a different device (e.g., GPU).
    self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )


    self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_head, dropout=drop_rate,  bias=False, kdim = self.kdim, vdim = self.vdim, batch_first = True)

    self.norm = nn.LayerNorm(hidden_dim)

    self.fc = nn.Linear(hidden_dim, items_size)

  def attention(self, lstm_hidden_layer):
    # convert to query, key and the value matrixes. 
    query = self.q(lstm_hidden_layer)
    key = self.k(lstm_hidden_layer)
    value = self.v(lstm_hidden_layer)
    # can input batch: because output is (N,L,E) when batch_first=True
    attention_out, weights = self.multihead_attention(query, key, value, attn_mask=self.mask)
    return attention_out
    

  def forward(self, x, h0=None, c0=None):

    # Initialize hidden state and cell state if not provided
    if h0 is None or c0 is None:
      h0 = torch.zeros(self.layer_dim, x.size(
          0), self.hidden_dim).to(x.device)
      c0 = torch.zeros(self.layer_dim, x.size(
          0), self.hidden_dim).to(x.device)


    embedded = self.embedding(x)

    lstm_out, (hn, cn) = self.lstm(embedded, (h0, c0))
    attended_out = self.attention(lstm_out)
    norm_out = self.norm(attended_out + lstm_out)
    output = self.fc(norm_out[:, -1, :])
    return output, hn, cn



class LSTMAttentionRec(nn.Module):
    def __init__(
        self, 
        items_size: int, 
        embedded_dim: int = 128, 
        hidden_dim: int = 128, 
        n_layers: int = 1, 
        n_heads: int = 2, 
        drop_rate: float = 0.1,
        pad_token_id: int = 0,
        tie_weights: bool = True,
        use_recency_bias: bool = True,
        recency_decay: float = 0.9,
        context_length: int = 10
    ) -> None:

        super(LSTMAttentionRec, self).__init__()
        self.pad_token_id = pad_token_id
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.tie_weights = tie_weights
        self.use_recency_bias = use_recency_bias
        self.recency_decay = float(recency_decay)
        self._max_len = context_length
        self.emb_dim = embedded_dim
        self.embedding = nn.Embedding(items_size, embedded_dim, padding_idx=pad_token_id)
        self.emb_drop = nn.Dropout(drop_rate)

        # Small init: the embedding matrix doubles as the output projection when
        # ``tie_weights`` is set, so an N(0, 1) default would blow up the logits.
        # The LSTM input path compensates with a sqrt(emb_dim) scale (see forward).
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[pad_token_id].zero_()


        lstm_dropout = drop_rate if n_layers > 1 else 0.0


        self.lstm = nn.LSTM(
            embedded_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            batch_first=True, 
            dropout=lstm_dropout
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

        # Tied output decoder 
        # If the hidden_dim != emb_dim, we need a linear layer to project the hidden states to the embedding dimension before applying the output projection.
        if tie_weights:
            self.h2e = nn.Linear(hidden_dim, embedded_dim, bias=False) if hidden_dim != embedded_dim else None
            self.decoder_bias = nn.Parameter(torch.zeros(items_size))
        else:
            self.fc = nn.Linear(hidden_dim, items_size)



        # Correct: Registering the tensor as a buffer. This ensures that the tensor is moved to the appropriate device when the model is moved to a different device (e.g., GPU).
        self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length),
                            diagonal=1), persistent=False
            )

    # ----------------------------------------------------------------------
    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len <= self._max_len:
            return self.mask[:seq_len, :seq_len].to(device)
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )

    # Recency bias: a decaying weight for each position in the sequence, applied to the attention scores before softmax. 
    # This encourages the model to pay more attention to recent items in the sequence.
    def _decode(self, h: torch.Tensor) -> torch.Tensor:
            if not self.tie_weights:
                return self.fc(h)
            if self.h2e is not None:
                h = self.h2e(h)
            return F.linear(h, self.embedding.weight, self.decoder_bias)

    @staticmethod
    def _infer_lengths(x: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        return (x != pad_token_id).sum(dim=1)
    


    # ----------------------------------------------------------------------

    def forward(self, x, h0=None, c0=None):
        """
            Args:
                x:       (B, L) right-padded token-id tensor.
                lengths: (B,) count of real tokens per row. Inferred (assuming
                            right-padding) when ``None``.
    
            Returns:
                logits:  (B, L, V) next-item logits at every position.
                (hn, cn): final LSTM states.
        """
        # Initialize batch size and sequence length
        bsz, seq_len = x.shape
        # if lengths is None:
        #     lengths = self._infer_lengths(x, self.pad_token_id)
        # lengths = lengths.clamp(min=1)

        # Initialize hidden state and cell state if not provided
        if h0 is None or c0 is None:

            h0 = torch.zeros(self.n_layers, x.size(
                0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.n_layers, x.size(
                0), self.hidden_dim).to(x.device)


        embedded = self.emb_drop(self.embedding(x))

        lstm_out, (hn, cn) = self.lstm(embedded, (h0, c0))

        # key_padding_mask = (
        #             torch.arange(seq_len, device=x.device)[None, :] >= self._max_len
        #         )   



        attn_out, _ = self.attn(
                    lstm_out,
                    lstm_out,
                    lstm_out,
                    attn_mask=self._causal_mask(seq_len, x.device),
                    #key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
        norm_out = self.norm(attn_out + lstm_out)
        norm_out= self.out_drop(norm_out)


        #output = self.fc(norm_out[:, -1, :])
        output = self._decode(norm_out[:, -1, :])  

        return output, hn, cn


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





class LSTMAttentionMetaEmbModel(nn.Module):

  def __init__(self, cfg, tokenizer=None):

    super(LSTMAttentionMetaEmbModel, self).__init__()

    self.drop_rate = cfg["drop_rate"] 
    self.hidden_dim = cfg["hidden_dim"]
    self.layer_dim = cfg["n_layers"]
    self.embedded_dim = cfg["emb_dim"]
    self.items_size = cfg["items_size"]
    self.contexlen = cfg["context_length"]
    self.kdim = cfg["hidden_dim"]
    self.vdim = cfg["hidden_dim"]
    self.n_heads = cfg["n_heads"]
    


    self.embedding = nn.Embedding(self.items_size,self.embedded_dim)

    self.use_meta = cfg["use_meta_embeddings"]
    if self.use_meta:
        assert tokenizer is not None, "tokenizer required when use_meta_embeddings=true"
        meta_emb, tok_to_meta = _build_meta_embedding(
            cfg["emb_dim"], tokenizer,
            cfg["item_meta_embedding"],
            cfg["item_meta_id_map"],
        )
        self.meta_emb = meta_emb
        if cfg.get("freeze_meta_embeddings", True):
            self.meta_emb.weight.requires_grad_(False)
        self.register_buffer("tok_to_meta", tok_to_meta)
        # Learned scalar gate, starts at 0 (no-op) — same trick used in
        # LSTMAttentionRec (src/lstm_v2.py) so the content prior only
        # contributes once training shows it helps.
        self.meta_gate = nn.Parameter(torch.zeros(()))


    if self.layer_dim > 1:
        self.lstm = nn.LSTM(self.embedded_dim, self.hidden_dim, num_layers=self.layer_dim, batch_first=True, dropout=self.drop_rate)
    else:
        self.lstm = nn.LSTM(self.embedded_dim, self.hidden_dim, num_layers=self.layer_dim, batch_first=True)

    # multihead attention
    self.q = nn.Linear(self.hidden_dim, self.kdim)
    self.k = nn.Linear(self.hidden_dim, self.kdim)
    self.v = nn.Linear(self.hidden_dim, self.vdim)

    # Correct: Registering the tensor as a buffer. This ensures that the tensor is moved to the appropriate device when the model is moved to a different device (e.g., GPU).
    self.register_buffer(
            "mask",
            torch.triu(torch.ones(self.contexlen , self.contexlen ),
                       diagonal=1)
        )


    self.multihead_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.n_heads, dropout=self.drop_rate,  bias=False, kdim = self.kdim, vdim = self.vdim, batch_first = True)

    self.norm = nn.LayerNorm(self.hidden_dim)

    self.fc = nn.Linear(self.hidden_dim, self.items_size)

  def attention(self, lstm_hidden_layer):
    # convert to query, key and the value matrixes. 
    query = self.q(lstm_hidden_layer)
    key = self.k(lstm_hidden_layer)
    value = self.v(lstm_hidden_layer)
    # can input batch: because output is (N,L,E) when batch_first=True
    attention_out, weights = self.multihead_attention(query, key, value, attn_mask=self.mask)
    return attention_out
    

  def forward(self, x, h0=None, c0=None):

    # Initialize hidden state and cell state if not provided
    if h0 is None or c0 is None:
      h0 = torch.zeros(self.layer_dim, x.size(
          0), self.hidden_dim).to(x.device)
      c0 = torch.zeros(self.layer_dim, x.size(
          0), self.hidden_dim).to(x.device)


    embedded = self.embedding(x)
    if self.use_meta:
        meta = self.meta_emb(self.tok_to_meta[x])
        embedded = embedded + self.meta_gate * meta

    lstm_out, (hn, cn) = self.lstm(embedded, (h0, c0))
    attended_out = self.attention(lstm_out)
    norm_out = self.norm(attended_out + lstm_out)
    output = self.fc(norm_out[:, -1, :])
    return output, hn, cn


