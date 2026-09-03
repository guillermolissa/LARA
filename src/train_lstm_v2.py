"""
train_lstm_v2.py — training loop for ``lstm_v2.LSTMAttentionRec``.

Applies the Tier-1 / Tier-2 training-side changes from the baseline review:

  * per-step cross-entropy over the shifted target sequence
    (``ignore_index = -100``) — every position is supervised, not just the
    last item;
  * ``label_smoothing`` and ``weight_decay`` read from config (recommended
    ~0.0-0.05 and ~0.01-0.05 respectively — far below the old 0.2 / 0.20);
  * linear-warmup + cosine LR decay (replaces ``CosineAnnealingWarmRestarts``);
  * gradient clipping (the original had it commented out);
  * **early stopping and checkpointing on validation NDCG@K**, not val loss.

Everything else (dataset, tokenizer, metrics, EarlyStopping, model naming)
is reused from the existing modules. K-fold is dropped in favour of a single
seeded holdout split for clarity — wrap the body in a ``KFold`` loop if you
need CV.

Usage:
    python src/train_lstm_v2.py --source dressipi
    python src/train_lstm_v2.py --config config/config.yaml --eval-k 20
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from functools import partial
from tqdm.auto import tqdm

import metrics
from custom_collate_v2 import IGNORE_INDEX, collate_fn_v2
from dataset import DressipiDataset
from lstm_v2 import LSTMAttentionRec
from tokenizer import get_tokenizer
from utils import EarlyStopping, build_model_name, load_config, set_seed

warnings.filterwarnings("ignore")

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[CLS]", "[SEP]", "[MASK]"]


def pick_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.05):
    """Linear warmup then cosine decay to ``min_ratio`` of the base LR."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, k: int, special_ids: list[int]) -> dict:
    """Ranking metrics @k on the validation split, using the last real step."""
    model.eval()
    gts: list[list[str]] = []
    preds: list[list[str]] = []

    for inputs, targets, lengths in loader:
        inputs, lengths = inputs.to(device), lengths.to(device)
        last_logits = model.predict_last(inputs, lengths, mask_token_ids=special_ids)
        top_ids = torch.topk(last_logits, k=k, dim=-1).indices.tolist()
        tgt_pos = (lengths - 1).tolist()

        for i, row in enumerate(top_ids):
            gt_id = int(targets[i, tgt_pos[i]].item())
            if gt_id == IGNORE_INDEX or gt_id in special_ids:
                continue
            gts.append([tokenizer.id_to_token(gt_id)])
            preds.append([tokenizer.id_to_token(t) for t in row])

    model.train()

    if not gts:
        return {n: float("nan") for n in ("hr", "mrr", "ndcg", "recall", "map", "precision")}

    pairs = list(zip(gts, preds))
    return {
        "hr": float(np.mean([metrics.hit_rate_k(g, p, k) for g, p in pairs])),
        "mrr": float(np.mean([metrics.rr_k(g, p, k) for g, p in pairs])),
        "ndcg": float(np.mean([metrics.ndcg_k(g, p, k) for g, p in pairs])),
        "recall": float(np.mean([metrics.recall_k(g, p, k) for g, p in pairs])),
        "map": float(np.mean([metrics.apk(g, p, k) for g, p in pairs])),
        "precision": float(np.mean([metrics.precision_k(g, p, k) for g, p in pairs])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train lstm_v2.LSTMAttentionRec")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--source", choices=["dressipi", "trivago", "spotify"], default=None)
    ap.add_argument("--eval-k", type=int, default=20)
    args = ap.parse_args()

    cfg_path = f"config/{args.source}.yaml" if args.source else args.config
    cfg = load_config(cfg_path)
    m, d, hp = cfg["model"], cfg["data"], cfg["hyperparameters"]

    set_seed(hp["seed"])
    device = pick_device(hp.get("device", "cpu"))
    print(f"config={cfg_path}  device={device}")

    tokenizer = get_tokenizer(tokenizer_path=d["tokenizer_path"])
    pad_id = tokenizer.token_to_id("[PAD]")
    special_ids = [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS
                   if tokenizer.token_to_id(t) is not None]
    m["items_size"] = tokenizer.get_vocab_size()

    train_file = Path(d["train"], d["train_feature_store"].rsplit(".", 1)[0] + ".parquet")
    dataset = DressipiDataset(file_path=train_file, tokenizer=tokenizer)

    n_val = int(len(dataset) * hp["validation_ratio"])
    n_train = len(dataset) - n_val
    split_gen = torch.Generator().manual_seed(hp["seed"])
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=split_gen)
    print(f"dataset={len(dataset):,}  train={n_train:,}  val={n_val:,}")

    collate = partial(collate_fn_v2, pad_token_id=pad_id, context_length=m["context_length"])
    num_workers = hp.get("num_workers") or 0
    train_dl = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,
                          drop_last=True, collate_fn=collate, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=hp["batch_size"], shuffle=False,
                        drop_last=False, collate_fn=collate, num_workers=num_workers)

    model = LSTMAttentionRec(
        items_size=m["items_size"],
        emb_dim=m["emb_dim"],
        hidden_dim=m["hidden_dim"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        drop_rate=m["drop_rate"],
        pad_token_id=pad_id,
        tie_weights=m.get("tie_weights", True),
        use_recency_bias=m.get("use_recency_bias", True),
        recency_decay=m.get("recency_decay", 0.9),
        max_len=m["context_length"] + 1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"]
    )
    epochs = hp["num_epochs"]
    total_steps = epochs * len(train_dl)
    scheduler = make_scheduler(optimizer, hp.get("warmup_steps", 0), total_steps)

    label_smoothing = hp.get("label_smoothing", 0.0)
    max_grad_norm = hp.get("max_grad_norm", 1.0)

    stopper = EarlyStopping(patience=hp.get("patience", 5), delta=0.0)
    ckpt_dir = Path(m["folder"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # ``_v2`` suffix so this does not clobber a v1 checkpoint of the same shape.
    model_name = build_model_name(m, hp).replace(".pth", "_v2.pth")
    best_path = ckpt_dir / model_name
    best_ndcg = -1.0

    print(f"label_smoothing={label_smoothing}  weight_decay={hp['weight_decay']}  "
          f"lr={hp['learning_rate']}  warmup={hp.get('warmup_steps', 0)}  "
          f"total_steps={total_steps}  early-stop on val NDCG@{args.eval_k}")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for inputs, targets, lengths in tqdm(train_dl, desc=f"epoch {epoch}", colour="green"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs, lengths)                       # (B, L, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                label_smoothing=label_smoothing,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        val = evaluate(model, val_dl, tokenizer, device, args.eval_k, special_ids)
        print(
            f"epoch {epoch:02d}  train_loss={running / len(train_dl):.4f}  "
            f"val NDCG@{args.eval_k}={val['ndcg']:.4f}  HR@{args.eval_k}={val['hr']:.4f}  "
            f"MRR@{args.eval_k}={val['mrr']:.4f}  Recall@{args.eval_k}={val['recall']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # EarlyStopping treats its arg as a loss (lower = better) → pass -NDCG.
        stopper(-val["ndcg"], model)
        if val["ndcg"] > best_ndcg:
            best_ndcg = val["ndcg"]
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ new best, saved → {best_path}  (NDCG@{args.eval_k}={best_ndcg:.4f})")

        if stopper.early_stop:
            print(f"early stopping at epoch {epoch}")
            break

    print(f"done. best val NDCG@{args.eval_k}={best_ndcg:.4f}  weights → {best_path}")


if __name__ == "__main__":
    main()
