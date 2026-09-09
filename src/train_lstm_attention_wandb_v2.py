"""
train_lstm_attention_wandb_v2.py — W&B-tracked training loop for
``lstm_v2.LSTMAttentionRec``.

This is the ``*_v2`` counterpart of ``train_lstm_attention_wandb.py``: it keeps
that script's Weights & Biases integration and checkpoint/resume machinery, but
swaps in the whole v2 pipeline:

  * model      → ``lstm_v2.LSTMAttentionRec`` (padding_idx, boolean causal mask,
                 key_padding_mask, per-step logits, weight tying, recency bias);
  * collate    → ``custom_collate.collate_train_fn`` (right-padding, shifted
                 per-step targets with ``IGNORE_INDEX``, returns ``lengths``);
  * loss       → per-step cross-entropy over the shifted target sequence
                 (every position supervised, not just the last item);
  * scheduler  → linear warmup + cosine decay (replaces
                 ``CosineAnnealingWarmRestarts``);
  * selection  → early stopping / checkpointing on **validation NDCG@K**, with
                 val loss and the full ranking-metric suite also logged to W&B.

K-fold is dropped in favour of a single seeded holdout split (as in
``train_lstm_v2.py``).

Usage:
    python src/train_lstm_attention_wandb_v2.py --source dressipi
    python src/train_lstm_attention_wandb_v2.py --config config/config.yaml --eval-k 20
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

import wandb

import metrics
from custom_collate import IGNORE_INDEX, collate_train_fn
from dataset import ItemDataset
from lstm_v2 import LSTMAttentionRec
from tokenizer import get_tokenizer
from utils import EarlyStopping, build_model_name, load_config, save_model, set_seed

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


def per_step_ce(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy over every (non-ignored) position of the shifted target."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        label_smoothing=label_smoothing,
    )


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, k: int, special_ids: list[int],
             label_smoothing: float = 0.0) -> dict:
    """Per-step val loss + ranking metrics @k (scored at the last real step)."""
    model.eval()
    gts: list[list[str]] = []
    preds: list[list[str]] = []
    loss_sum, n_batches = 0.0, 0

    for inputs, targets, lengths in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        logits, _ = model(inputs, lengths)
        loss_sum += per_step_ce(logits, targets, label_smoothing).item()
        n_batches += 1

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

    val_loss = loss_sum / max(1, n_batches)
    if not gts:
        out = {n: float("nan") for n in ("hr", "mrr", "ndcg", "recall", "map", "precision")}
        out["loss"] = val_loss
        return out

    pairs = list(zip(gts, preds))
    return {
        "loss": val_loss,
        "hr": float(np.mean([metrics.hit_rate_k(g, p, k) for g, p in pairs])),
        "mrr": float(np.mean([metrics.rr_k(g, p, k) for g, p in pairs])),
        "ndcg": float(np.mean([metrics.ndcg_k(g, p, k) for g, p in pairs])),
        "recall": float(np.mean([metrics.recall_k(g, p, k) for g, p in pairs])),
        "map": float(np.mean([metrics.apk(g, p, k) for g, p in pairs])),
        "precision": float(np.mean([metrics.precision_k(g, p, k) for g, p in pairs])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="W&B training for lstm_v2.LSTMAttentionRec")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--source", choices=["dressipi", "trivago", "spotify"], default=None)
    ap.add_argument("--eval-k", type=int, default=20)
    args = ap.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    cfg_path = f"config/{args.source}.yaml" if args.source else args.config
    cfg = load_config(cfg_path)
    m, d, hp, ex = cfg["model"], cfg["data"], cfg["hyperparameters"], cfg["experiment"]

    if ex.get("wandb_api_key") and ex["wandb_api_key"] != "api-key-goes-here":
        os.environ["WANDB_API_KEY"] = ex["wandb_api_key"]

    set_seed(hp["seed"])
    device = pick_device(hp.get("device", "cpu"))
    num_workers = hp.get("num_workers") or 0
    print(f"config={cfg_path}  device={device}")

    tokenizer = get_tokenizer(tokenizer_path=d["tokenizer_path"])
    pad_id = tokenizer.token_to_id("[PAD]")
    special_ids = [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS
                   if tokenizer.token_to_id(t) is not None]
    m["items_size"] = tokenizer.get_vocab_size()

    # --- data --------------------------------------------------------------
    train_file = Path(d["train"], d["train_feature_store"].rsplit(".", 1)[0] + ".parquet")
    dataset = ItemDataset(file_path=train_file, tokenizer=tokenizer)

    n_val = int(len(dataset) * hp["validation_ratio"])
    n_train = len(dataset) - n_val
    split_gen = torch.Generator().manual_seed(hp["seed"])
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=split_gen)
    print(f"dataset={len(dataset):,}  train={n_train:,}  val={n_val:,}")

    collate = partial(collate_train_fn, pad_token_id=pad_id, context_length=m["context_length"])
    train_dl = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,
                          drop_last=True, collate_fn=collate, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=hp["batch_size"], shuffle=False,
                        drop_last=False, collate_fn=collate, num_workers=num_workers)

    # --- model / optim ---------------------------------------------------
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

    early_stopping = EarlyStopping(patience=hp.get("patience", 5), delta=0.0)

    ckpt_dir = Path(m["folder"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_name = build_model_name(m, hp).replace(".pth", "_v2.pth")
    best_path = ckpt_dir / model_name
    checkpoint_path = ckpt_dir / model_name.replace(".pth", "_checkpoint.pth")

    # --- resume --------------------------------------------------------
    start_epoch = 0
    best_ndcg = -1.0
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_ndcg = ckpt.get("best_ndcg", -1.0)
        print(f"checkpoint loaded — resuming from epoch {start_epoch} (best NDCG@{args.eval_k}={best_ndcg:.4f})")

    # --- wandb config -----------------------------------------------
    wb_config = dict(m) | dict(hp)
    wb_config.update(
        train_feature_store=d["train_feature_store"],
        config_path=cfg_path,
        collate="collate_train_fn",
        model_class="LSTMAttentionRec",
        loss_fn="per_step_cross_entropy",
        scheduler="warmup_cosine",
        optimizer=optimizer.__class__.__name__,
        selection_metric=f"val_ndcg@{args.eval_k}",
        model_name=model_name,
        n_params=n_params,
    )

    print(f"label_smoothing={label_smoothing}  weight_decay={hp['weight_decay']}  "
          f"lr={hp['learning_rate']}  warmup={hp.get('warmup_steps', 0)}  "
          f"total_steps={total_steps}  early-stop on val NDCG@{args.eval_k}")

    run = wandb.init(
        entity=ex.get("entity"),
        project=ex.get("project"),
        id=ex.get("run_id"),
        resume=ex.get("resume", "allow"),
        group=ex.get("group"),
        notes=ex.get("notes"),
        config=wb_config,
    )

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            running = 0.0
            for inputs, targets, lengths in tqdm(
                train_dl, desc=f"epoch {epoch + 1}/{epochs}", colour="green"
            ):
                inputs = inputs.to(device)
                targets = targets.to(device)
                lengths = lengths.to(device)

                optimizer.zero_grad()
                logits, _ = model(inputs, lengths)                   # (B, L, V)
                loss = per_step_ce(logits, targets, label_smoothing)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                running += loss.item()

            train_loss = running / len(train_dl)
            val = evaluate(model, val_dl, tokenizer, device, args.eval_k,
                           special_ids, label_smoothing)
            lr_now = scheduler.get_last_lr()[0]

            run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val["loss"],
                f"val/ndcg@{args.eval_k}": val["ndcg"],
                f"val/hr@{args.eval_k}": val["hr"],
                f"val/mrr@{args.eval_k}": val["mrr"],
                f"val/recall@{args.eval_k}": val["recall"],
                f"val/map@{args.eval_k}": val["map"],
                f"val/precision@{args.eval_k}": val["precision"],
                "lr": lr_now,
                "recency_gamma": (
                    float(model.recency_gamma.detach().cpu())
                    if getattr(model, "use_recency_bias", False) else None
                ),
            })

            print(
                f"epoch {epoch + 1:02d}  train_loss={train_loss:.4f}  "
                f"val_loss={val['loss']:.4f}  val NDCG@{args.eval_k}={val['ndcg']:.4f}  "
                f"HR@{args.eval_k}={val['hr']:.4f}  MRR@{args.eval_k}={val['mrr']:.4f}  "
                f"Recall@{args.eval_k}={val['recall']:.4f}  lr={lr_now:.2e}"
            )

            if val["ndcg"] > best_ndcg:
                best_ndcg = val["ndcg"]
                torch.save(model.state_dict(), best_path)
                run.summary[f"best_val_ndcg@{args.eval_k}"] = best_ndcg
                print(f"  ↳ new best, saved → {best_path}  (NDCG@{args.eval_k}={best_ndcg:.4f})")

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_ndcg": best_ndcg,
                },
                checkpoint_path,
            )

            # EarlyStopping treats its arg as a loss (lower = better) → pass -NDCG.
            early_stopping(-val["ndcg"], model)
            if early_stopping.early_stop:
                print(f"early stopping at epoch {epoch + 1}  "
                      f"(best NDCG@{args.eval_k}={early_stopping.best_score:.4f})")
                break
    finally:
        run.finish()

    print(f"done. best val NDCG@{args.eval_k}={best_ndcg:.4f}  weights → {best_path}")


if __name__ == "__main__":
    main()
