"""
build_item_embeddings.py
------------------------
Trains a self-supervised item embedding model from sparse item metadata
(feature_category_id / feature_value_id pairs) and saves the resulting
dense embedding matrix for use as pretrained item content features in
``LSTMAttentionRec`` (see ``src/lstm_v2.py``) or the LARA transformer.

Architecture overview
---------------------
1. For each item, every (category, value) pair is encoded as:
       feat_emb = cat_emb + val_emb   (element-wise sum, both R^d)
   Summing keeps the same output dimension while letting the model learn
   which categories and values shift the representation. Features are
   sorted by ``feature_category_id`` so every item presents its features
   to the encoder in the same canonical order — required for the LSTM
   below to see a consistent "position" per category across items, since
   the raw CSV row order is arbitrary.

2. All per-feature embeddings for one item are aggregated into a single
   fixed-size vector via an **LSTM + self-attention encoder**, the same
   pattern ``LSTMAttentionRec`` uses for session sequences:
       feat_emb  --BiLSTM-->  contextualised features
                 --self-attention + residual + LayerNorm-->  refined features
                 --attention pooling (learnable query)-->  item vector
   The bidirectional LSTM lets each feature's representation absorb
   context from every other feature of the item (e.g. "color=blue" reads
   differently on a "category=shirt" item vs. a "category=shoes" item);
   self-attention then lets features re-weigh each other directly/without
   the LSTM's recency bias; only a final learnable-query attention pool
   collapses the (variable-length, order-insensitive) feature set into a
   fixed-size vector, so the arbitrary-but-consistent LSTM ordering never
   leaks into the pooled output as a hard positional bias. This is a
   deliberate step up from plain mean/attention pooling over raw feature
   embeddings: it captures *interactions* between an item's features, not
   just their (weighted) sum.

3. Training objective — feature reconstruction (self-supervised):
   For every (category, value) pair in an item, the model is asked to
   predict feature_value_id from (item_embedding + category_embedding).
   Cross-entropy loss forces the compressed item vector to retain enough
   information to reconstruct all its features, yielding semantically
   rich dense representations without any session-level labels.

Integration into LSTMAttentionRec
----------------------------------
Load the saved embeddings at model initialisation:
    meta_emb = nn.Embedding.from_pretrained(torch.load("item_meta_embeddings.pt"))
Then combine with item-ID embeddings inside the recurrent model's input
layer (see ``use_meta_embeddings`` support added to ``LSTMAttentionRec``
in ``src/lstm_v2.py``):
    token = id_emb(item_ids) + meta_emb(item_ids)
The meta embedding acts as a content prior that biases ID embeddings
toward semantically similar neighbours from the start of training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # I/O
    data_path: str = "data/raw/item_features.csv"
    output_dir: str = "data/processed"
    emb_file: str = "item_meta_embeddings.pt"
    map_file: str = "item_id_to_index.json"

    # Model
    embed_dim: int = 128        # output embedding dimension (match Transformer d_model)
    lstm_hidden_dim: int = 64   # per-direction LSTM hidden size (bidirectional -> 2x, projected back to embed_dim)
    lstm_layers: int = 1        # number of stacked LSTM layers over the feature sequence
    num_heads: int = 4          # attention heads (self-attention + pooling layers)
    dropout: float = 0.1

    # Training
    batch_size: int = 512
    lr: float = 1e-3
    num_epochs: int = 30
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    seed: int = 42

    # Logging
    log_every: int = 50         # log every N batches

    def __post_init__(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

@dataclass
class FeatureData:
    """Holds re-indexed feature tensors and vocabulary sizes."""
    # per-item: list of (category_idx, value_idx) pairs
    item_features: Dict[int, Tuple[List[int], List[int]]]
    num_categories: int
    num_values: int
    item_id_to_index: Dict[int, int]  # original item_id -> contiguous index
    index_to_item_id: Dict[int, int]


def load_and_index(data_path: str) -> FeatureData:
    """Read CSV and build contiguous integer indices for all IDs."""
    df = pd.read_csv(data_path)
    logging.info("Loaded %d rows, %d unique items", len(df), df["item_id"].nunique())

    # Build contiguous indices (leave 0 as padding sentinel)
    cat_ids = sorted(df["feature_category_id"].unique())
    val_ids = sorted(df["feature_value_id"].unique())
    item_ids = sorted(df["item_id"].unique())

    cat_to_idx: Dict[int, int] = {c: i + 1 for i, c in enumerate(cat_ids)}  # 1-based
    val_to_idx: Dict[int, int] = {v: i + 1 for i, v in enumerate(val_ids)}
    item_id_to_index: Dict[int, int] = {it: i for i, it in enumerate(item_ids)}
    index_to_item_id: Dict[int, int] = {i: it for it, i in item_id_to_index.items()}

    df["cat_idx"] = df["feature_category_id"].map(cat_to_idx)
    df["val_idx"] = df["feature_value_id"].map(val_to_idx)

    # Canonical per-item ordering: the LSTM aggregator (see LSTMAttentionContextualizer)
    # needs a consistent feature order across items — raw CSV row order is arbitrary.
    df = df.sort_values(["item_id", "cat_idx"])

    item_features: Dict[int, Tuple[List[int], List[int]]] = {}
    for item_id, grp in df.groupby("item_id"):
        idx = item_id_to_index[item_id]
        item_features[idx] = (
            grp["cat_idx"].tolist(),
            grp["val_idx"].tolist(),
        )

    return FeatureData(
        item_features=item_features,
        num_categories=len(cat_ids) + 1,   # +1 for padding idx 0
        num_values=len(val_ids) + 1,
        item_id_to_index=item_id_to_index,
        index_to_item_id=index_to_item_id,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ItemFeatureDataset(Dataset):
    """
    Each sample is one item. Returns:
        item_idx   : scalar int
        cat_idxs   : LongTensor [n_feat]
        val_idxs   : LongTensor [n_feat]
    """

    def __init__(self, feature_data: FeatureData) -> None:
        self.items = list(feature_data.item_features.keys())
        self.features = feature_data.item_features

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, pos: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        idx = self.items[pos]
        cats, vals = self.features[idx]
        return (
            idx,
            torch.tensor(cats, dtype=torch.long),
            torch.tensor(vals, dtype=torch.long),
        )


def collate_fn(
    batch: List[Tuple[int, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad variable-length feature sequences to the max length in the batch.
    Returns:
        item_idxs   [B]
        cat_padded  [B, L]
        val_padded  [B, L]
        pad_mask    [B, L]  True where padding (for nn.MultiheadAttention key_padding_mask)
    """
    item_idxs, cats, vals = zip(*batch)
    max_len = max(c.size(0) for c in cats)
    B = len(item_idxs)

    cat_padded = torch.zeros(B, max_len, dtype=torch.long)
    val_padded = torch.zeros(B, max_len, dtype=torch.long)
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)   # True = ignore

    for i, (c, v) in enumerate(zip(cats, vals)):
        n = c.size(0)
        cat_padded[i, :n] = c
        val_padded[i, :n] = v
        pad_mask[i, :n] = False   # real tokens are NOT masked

    return (
        torch.tensor(item_idxs, dtype=torch.long),
        cat_padded,
        val_padded,
        pad_mask,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """
    Aggregates a variable-length sequence of feature embeddings into one
    fixed-size vector using cross-attention with a single learnable query.

    The query learns which feature patterns matter for a general-purpose
    item representation, independent of feature order or count.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        feat_embs: torch.Tensor,      # [B, L, d]
        key_padding_mask: torch.Tensor,  # [B, L]  True = padding
    ) -> torch.Tensor:                # [B, d]
        B = feat_embs.size(0)
        query = self.query.expand(B, -1, -1)            # [B, 1, d]
        out, _ = self.attn(
            query, feat_embs, feat_embs,
            key_padding_mask=key_padding_mask,
        )                                                # [B, 1, d]
        return self.norm(out.squeeze(1))                 # [B, d]


class LSTMAttentionContextualizer(nn.Module):
    """
    Contextualises a variable-length feature sequence with a bidirectional
    LSTM followed by self-attention — the same LSTM -> attention -> residual
    -> LayerNorm block ``LSTMAttentionRec`` (src/lstm_v2.py) applies to
    session sequences, reused here so each feature embedding absorbs
    information from every other feature belonging to the same item before
    pooling.

    Unlike ``LSTMAttentionRec`` this is a *set* encoder, not autoregressive:
    the LSTM is bidirectional and self-attention uses no causal mask, only
    a key-padding mask, since there is no "future" to hide.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project the concatenated forward/backward hidden states back to embed_dim
        # so the residual self-attention block below operates in the same space
        # as the raw feature embeddings (mirrors LSTMAttentionRec's h2e tying trick).
        self.proj = nn.Linear(2 * hidden_dim, embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        feat_emb: torch.Tensor,          # [B, L, d]
        lengths: torch.Tensor,           # [B]  real (non-padded) feature count
        key_padding_mask: torch.Tensor,  # [B, L]  True = padding
    ) -> torch.Tensor:                   # [B, L, d]
        # pack/pad so the LSTM ignores padding positions entirely (no wasted
        # compute, and padded rows can't leak zero-vectors into real states).
        packed = nn.utils.rnn.pack_padded_sequence(
            feat_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=feat_emb.size(1)
        )                                                    # [B, L, 2*hidden_dim]
        lstm_out = self.proj(lstm_out)                        # [B, L, d]

        attn_out, _ = self.attn(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )                                                      # [B, L, d]
        return self.norm(attn_out + lstm_out)                  # [B, L, d]


class ItemMetaEncoder(nn.Module):
    """
    Encodes item metadata feature pairs (category_id, value_id) into a
    single dense embedding vector per item.

    Forward pass → item_embedding  [B, embed_dim]

    The reconstruction decoder (feature_value_logits) is only used during
    training and is discarded when saving embeddings.
    """

    def __init__(
        self,
        num_categories: int,
        num_values: int,
        embed_dim: int,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Separate embedding tables for category IDs and value IDs
        self.cat_emb = nn.Embedding(num_categories, embed_dim, padding_idx=0)
        self.val_emb = nn.Embedding(num_values, embed_dim, padding_idx=0)

        # LSTM + self-attention: let features of the same item interact
        # before they are pooled (see LSTMAttentionContextualizer docstring).
        self.context = LSTMAttentionContextualizer(
            embed_dim, lstm_hidden_dim, lstm_layers, num_heads, dropout
        )

        # Aggregate the contextualised per-feature embeddings → single item vector
        self.pool = AttentionPooling(embed_dim, num_heads, dropout)

        # Optional projection after pooling (adds expressivity)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

        # Reconstruction head: (item_emb + cat_emb) → value_id logits
        # Predicting per-category value forces item_emb to encode all features.
        self.value_decoder = nn.Linear(embed_dim, num_values)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.cat_emb.weight, std=0.02)
        nn.init.normal_(self.val_emb.weight, std=0.02)
        nn.init.zeros_(self.cat_emb.weight[0])   # padding idx stays zero
        nn.init.zeros_(self.val_emb.weight[0])

    def encode(
        self,
        cat_idxs: torch.Tensor,       # [B, L]
        val_idxs: torch.Tensor,       # [B, L]
        pad_mask: torch.Tensor,       # [B, L]
    ) -> torch.Tensor:                # [B, d]
        """Encode item features into a single dense vector."""
        feat_emb = self.cat_emb(cat_idxs) + self.val_emb(val_idxs)  # [B, L, d]
        lengths = (~pad_mask).sum(dim=1)                              # [B]
        feat_emb = self.context(feat_emb, lengths, pad_mask)          # [B, L, d]
        item_emb = self.pool(feat_emb, pad_mask)                      # [B, d]
        return self.proj(item_emb)                                    # [B, d]

    def forward(
        self,
        cat_idxs: torch.Tensor,   # [B, L]
        val_idxs: torch.Tensor,   # [B, L]
        pad_mask: torch.Tensor,   # [B, L]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            item_emb  [B, d]         — the dense item embedding
            logits    [B*L_real, V]  — value prediction logits (unpadded)
            targets   [B*L_real]     — true value indices (unpadded)
        """
        item_emb = self.encode(cat_idxs, val_idxs, pad_mask)  # [B, d]

        # Reconstruction: for each real (non-padded) feature position,
        # predict value_id from item_emb + category_emb.
        real_mask = ~pad_mask                               # [B, L]  True = real
        B, L = cat_idxs.shape

        # Gather the item embedding for each feature position
        item_exp = item_emb.unsqueeze(1).expand(-1, L, -1)     # [B, L, d]
        cat_exp = self.cat_emb(cat_idxs)                        # [B, L, d]
        query = item_exp + cat_exp                              # [B, L, d]

        logits_all = self.value_decoder(query)                  # [B, L, V]

        # Flatten and filter out padding positions
        logits_flat = logits_all.view(B * L, -1)               # [B*L, V]
        mask_flat = real_mask.view(B * L)                       # [B*L]
        targets_flat = val_idxs.view(B * L)                     # [B*L]

        logits = logits_flat[mask_flat]                         # [N_real, V]
        targets = targets_flat[mask_flat]                       # [N_real]

        return item_emb, logits, targets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup then cosine annealing."""
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.num_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(cfg: Config) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    torch.manual_seed(cfg.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    feature_data = load_and_index(cfg.data_path)
    dataset = ItemFeatureDataset(feature_data)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
    )
    logging.info(
        "Dataset: %d items | %d categories | %d values",
        len(dataset),
        feature_data.num_categories,
        feature_data.num_values,
    )

    # ------------------------------------------------------------------
    # Model, optimiser, scheduler
    # ------------------------------------------------------------------
    model = ItemMetaEncoder(
        num_categories=feature_data.num_categories,
        num_values=feature_data.num_values,
        embed_dim=cfg.embed_dim,
        lstm_hidden_dim=cfg.lstm_hidden_dim,
        lstm_layers=cfg.lstm_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(device)
    logging.info(
        "Model parameters: %s",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg, len(loader))
    criterion = nn.CrossEntropyLoss(ignore_index=0)   # index 0 is padding

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_loss = float("inf")
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (item_idxs, cat_padded, val_padded, pad_mask) in enumerate(loader, 1):
            cat_padded = cat_padded.to(device)
            val_padded = val_padded.to(device)
            pad_mask = pad_mask.to(device)

            _, logits, targets = model(cat_padded, val_padded, pad_mask)
            loss = criterion(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            if step % cfg.log_every == 0:
                avg = epoch_loss / step
                lr_now = scheduler.get_last_lr()[0]
                logging.info(
                    "Epoch %d/%d  step %d/%d  loss=%.4f  lr=%.2e",
                    epoch, cfg.num_epochs, step, len(loader), avg, lr_now,
                )

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        logging.info(
            "==> Epoch %d done  avg_loss=%.4f  time=%.1fs",
            epoch, avg_loss, elapsed,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_checkpoint(model, cfg, feature_data, device, tag="best")

    # ------------------------------------------------------------------
    # Extract and save final embeddings
    # ------------------------------------------------------------------
    _save_embeddings(model, cfg, feature_data, dataset, loader, device)


# ---------------------------------------------------------------------------
# Inference / embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_embeddings(
    model: ItemMetaEncoder,
    dataset: ItemFeatureDataset,
    feature_data: FeatureData,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Pass all items through the encoder and collect [num_items, d] matrix."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=min(4, os.cpu_count() or 1),
    )

    num_items = len(dataset)
    emb_dim = model.embed_dim
    # Pre-allocate output — items may arrive out of order due to DataLoader
    embedding_matrix = torch.zeros(num_items, emb_dim)

    for item_idxs, cat_padded, val_padded, pad_mask in loader:
        cat_padded = cat_padded.to(device)
        val_padded = val_padded.to(device)
        pad_mask = pad_mask.to(device)

        embs = model.encode(cat_padded, val_padded, pad_mask).cpu()  # [B, d]
        embedding_matrix[item_idxs] = embs

    return embedding_matrix


def _save_checkpoint(
    model: ItemMetaEncoder,
    cfg: Config,
    feature_data: FeatureData,
    device: torch.device,
    tag: str = "best",
) -> None:
    path = Path(cfg.output_dir) / f"item_meta_encoder_{tag}.pt"
    torch.save(model.state_dict(), path)
    logging.info("Saved checkpoint -> %s", path)


def _save_embeddings(
    model: ItemMetaEncoder,
    cfg: Config,
    feature_data: FeatureData,
    dataset: ItemFeatureDataset,
    loader: DataLoader,
    device: torch.device,
) -> None:
    logging.info("Extracting item embeddings ...")
    emb_matrix = _extract_embeddings(model, dataset, feature_data, device, cfg.batch_size)

    emb_path = Path(cfg.output_dir) / cfg.emb_file
    map_path = Path(cfg.output_dir) / cfg.map_file

    torch.save(emb_matrix, emb_path)
    logging.info("Saved embedding matrix %s -> %s", list(emb_matrix.shape), emb_path)

    # Save item_id → index map as JSON (human-readable + easy to reload)
    with open(map_path, "w") as f:
        json.dump(
            {str(k): v for k, v in feature_data.item_id_to_index.items()},
            f, indent=2,
        )
    logging.info("Saved id→index map (%d entries) -> %s", len(feature_data.item_id_to_index), map_path)

    # Quick sanity check: cosine similarity of first few items
    sample = emb_matrix[:5]
    norms = sample / sample.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim = norms @ norms.T
    logging.info("Cosine similarity (first 5 items):\n%s", sim.numpy().round(3))


# ---------------------------------------------------------------------------
# Argparse entry-point
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train item metadata embeddings for Dressiformer."
    )
    parser.add_argument("--data_path", default="data/raw/item_features.csv")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--emb_file", default="item_meta_embeddings.pt")
    parser.add_argument("--map_file", default="item_id_to_index.json")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--lstm_hidden_dim", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
