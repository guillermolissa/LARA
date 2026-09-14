"""
train_cv_lstm_attention_v2.py — K-fold CV training loop for
``lstm_v2.LSTMAttentionRec``.

This is the ``*_v2`` counterpart of ``train_cv_lstm_attention.py``: it keeps
that script's K-fold cross-validation and W&B/checkpoint-per-fold machinery,
but swaps in the v2 pipeline (see ``lstm_v2.py`` for the full rationale):

  * model      → ``lstm_v2.LSTMAttentionRec`` (padding_idx, boolean causal
                 mask, key_padding_mask, per-step logits, weight tying,
                 recency bias);
  * collate    → ``custom_collate.collate_next_item_fn`` (right-padding, shifted
                 per-step targets with ``IGNORE_INDEX``, returns ``lengths``);
  * loss       → per-step cross-entropy over the shifted target sequence
                 (every position supervised, not just the last item);
  * scheduler  → linear warmup + cosine decay (replaces
                 ``CosineAnnealingWarmRestarts``);
  * selection  → early stopping / checkpointing on **validation NDCG@K** per
                 fold, instead of val loss.

Adaptive softmax is dropped (not implemented by ``LSTMAttentionRec``); use
``train_cv_lstm_attention.py`` for that.

Usage:
    python src/train_cv_lstm_attention_v2.py --source dressipi
    python src/train_cv_lstm_attention_v2.py --source dressipi --wandb --eval-k 20
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb

import metrics
from custom_collate import IGNORE_INDEX, collate_next_item_fn
from dataset import ItemDataset
from lstm_v2 import LSTMAttentionRec
from tokenizer import get_tokenizer
from tokenizers import Tokenizer
from utils import EarlyStopping, build_model_name, load_config, set_seed

warnings.filterwarnings("ignore")

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[CLS]", "[SEP]", "[MASK]"]


def load_data(file_path: str, tokenizer: Tokenizer) -> ItemDataset:
    print("Loading dataset...")
    # It only has the train split, so we divide it ourselves (per fold, below).
    return ItemDataset(file_path=file_path, tokenizer=tokenizer)


def reset_weights(m: torch.nn.Module) -> None:
    """Resets trainable parameters of every child layer, to avoid weight
    leakage between CV folds."""
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()


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
def calc_loss_loader(loader: DataLoader, model: torch.nn.Module, device: torch.device,
                     label_smoothing: float = 0.0, num_batches: int | None = None) -> float:
    if len(loader) == 0:
        return float("nan")
    num_batches = len(loader) if num_batches is None else min(num_batches, len(loader))

    total_loss = 0.0
    for i, (inputs, targets, lengths) in enumerate(loader):
        if i >= num_batches:
            break
        inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
        logits, _ = model(inputs, lengths)
        total_loss += per_step_ce(logits, targets, label_smoothing).item()
    return total_loss / num_batches


def validate_model(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   device: torch.device, label_smoothing: float, num_batches: int | None) -> Tuple[float, float]:
    """Per-step CE loss on (a sample of) the train and val splits."""
    model.eval()
    train_loss = calc_loss_loader(train_loader, model, device, label_smoothing, num_batches)
    val_loss = calc_loss_loader(val_loader, model, device, label_smoothing, num_batches)
    model.train()
    return train_loss, val_loss


@torch.no_grad()
def evaluate_ranking_metrics(model: torch.nn.Module, val_loader: DataLoader, tokenizer: Tokenizer,
                             device: torch.device, k: int, special_ids: list[int]) -> Dict[str, float]:
    """Ranking metrics @k on the validation split, scored at the last real step."""
    model.eval()
    gts: list[list[str]] = []
    preds: list[list[str]] = []

    for inputs, targets, lengths in val_loader:
        inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
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

    names = ("hr", "mrr", "ndcg", "recall", "map", "precision")
    if not gts:
        return {f"{n}_{k}": float("nan") for n in names}

    pairs = list(zip(gts, preds))
    return {
        f"hr_{k}": float(np.mean([metrics.hit_rate_k(g, p, k) for g, p in pairs])),
        f"mrr_{k}": float(np.mean([metrics.rr_k(g, p, k) for g, p in pairs])),
        f"ndcg_{k}": float(np.mean([metrics.ndcg_k(g, p, k) for g, p in pairs])),
        f"recall_{k}": float(np.mean([metrics.recall_k(g, p, k) for g, p in pairs])),
        f"map_{k}": float(np.mean([metrics.apk(g, p, k) for g, p in pairs])),
        f"precision_{k}": float(np.mean([metrics.precision_k(g, p, k) for g, p in pairs])),
    }


def train_cv(cfg: dict, track_experiment: bool, verbose: bool, eval_k: int) -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    cfg_model, cfg_data, cfg_hp, cfg_ex = cfg["model"], cfg["data"], cfg["hyperparameters"], cfg["experiment"]

    if track_experiment and cfg_ex.get("wandb_api_key") and cfg_ex["wandb_api_key"] != "api-key-goes-here":
        os.environ["WANDB_API_KEY"] = cfg_ex["wandb_api_key"]

    num_workers = cfg_hp.get("num_workers") or 0
    device = pick_device(cfg_hp.get("device", "cpu"))
    print(f"device={device}")

    set_seed(cfg_hp["seed"])
    k_folds = cfg_hp["kfold"]

    file_path = Path(cfg_data["train"], cfg_data["train_feature_store"].rsplit(".", 1)[0] + ".parquet")
    tokenizer = get_tokenizer(tokenizer_path=cfg_data["tokenizer_path"])
    pad_id = tokenizer.token_to_id("[PAD]")
    special_ids = [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS
                   if tokenizer.token_to_id(t) is not None]

    seq_len = cfg_model["context_length"]
    batch_size = cfg_hp["batch_size"]
    cfg_model["items_size"] = tokenizer.get_vocab_size()

    # ``_v2`` suffix so this does not clobber a v1 checkpoint of the same shape.
    model_filename = build_model_name(cfg_model, cfg_hp).replace(".pth", "_v2.pth")
    ckpt_dir = Path(cfg_model["folder"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_data(file_path=file_path, tokenizer=tokenizer)
    kfold = KFold(n_splits=k_folds, shuffle=True)

    epochs = cfg_hp["num_epochs"]
    evaluation_num_batches = None
    warmup_steps = cfg_hp.get("warmup_steps", 0)
    max_grad_norm = cfg_hp.get("max_grad_norm", 1.0)
    label_smoothing = cfg_hp.get("label_smoothing", 0.0)

    collate = partial(collate_next_item_fn, pad_token_id=pad_id, context_length=seq_len)

    GROUP = cfg_ex["group"] + wandb.util.generate_id()

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold}")
        print("--------------------------------" * 2)

        model = LSTMAttentionRec(
            items_size=cfg_model["items_size"],
            emb_dim=cfg_model["emb_dim"],
            hidden_dim=cfg_model["hidden_dim"],
            n_layers=cfg_model["n_layers"],
            n_heads=cfg_model["n_heads"],
            drop_rate=cfg_model["drop_rate"],
            pad_token_id=pad_id,
            tie_weights=cfg_model.get("tie_weights", True),
            use_recency_bias=cfg_model.get("use_recency_bias", True),
            recency_decay=cfg_model.get("recency_decay", 0.9),
            max_len=seq_len + 1,
        ).to(device)

        if cfg_hp["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_hp["learning_rate"],
                                          weight_decay=cfg_hp["weight_decay"])
        elif cfg_hp["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg_hp["learning_rate"],
                                        weight_decay=cfg_hp["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer: {cfg_hp['optimizer']}")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler,
                                      collate_fn=collate, drop_last=True, num_workers=num_workers)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler,
                                    collate_fn=collate, drop_last=False, num_workers=num_workers)

        total_steps = epochs * len(train_dataloader)
        scheduler = make_scheduler(optimizer, warmup_steps, total_steps)

        checkpoint_path = ckpt_dir / model_filename.replace(".pth", f"_checkpoint_f{fold}.pth")
        best_path = ckpt_dir / model_filename.replace(".pth", f"_best_f{fold}.pth")
        early_stopping = EarlyStopping(patience=cfg_hp.get("patience", 5), delta=0.0)

        # config to log to wandb
        config = dict(cfg_model) | dict(cfg_hp)
        config.update(
            train_feature_store=cfg_data["train_feature_store"],
            model_name=model_filename,
            model_class="LSTMAttentionRec",
            collate="collate_next_item_fn",
            loss_fn="per_step_cross_entropy",
            scheduler="warmup_cosine",
            selection_metric=f"val_ndcg@{eval_k}",
            fold=fold,
        )

        run_name = f"fold-{fold}"
        run = None
        if track_experiment:
            run = wandb.init(entity=cfg_ex["entity"], project=cfg_ex["project"],
                             resume=cfg_ex["resume"], group=GROUP, name=run_name,
                             job_type=run_name, reinit=True, config=config)
            print("RUN WANDB INFO\n")
            print("ENTITY: ", cfg_ex["entity"], " - PROJECT: ", cfg_ex["project"], " - GROUP: ", GROUP)

        start_epoch = 0
        best_ndcg = -1.0

        model.apply(reset_weights)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_ndcg = checkpoint.get("best_ndcg", -1.0)
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"Checkpoint loaded. Resuming training from epoch {start_epoch} "
                  f"(best NDCG@{eval_k}={best_ndcg:.4f})")

        model.train()

        for epoch in tqdm(range(start_epoch, epochs), total=epochs, desc="Training...", colour="orange"):
            total_loss = 0.0

            for inputs, targets, lengths in train_dataloader:
                inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)

                optimizer.zero_grad()
                logits, _ = model(inputs, lengths)
                loss = per_step_ce(logits, targets, label_smoothing)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            train_loss, val_loss = validate_model(
                model, train_dataloader, val_dataloader, device,
                label_smoothing=label_smoothing, num_batches=evaluation_num_batches,
            )
            val_ranking_metrics = evaluate_ranking_metrics(
                model, val_dataloader, tokenizer, device, k=eval_k, special_ids=special_ids,
            )
            lr_now = scheduler.get_last_lr()[0]

            if track_experiment:
                run.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    f"val/ndcg@{eval_k}": val_ranking_metrics[f"ndcg_{eval_k}"],
                    f"val/hr@{eval_k}": val_ranking_metrics[f"hr_{eval_k}"],
                    f"val/mrr@{eval_k}": val_ranking_metrics[f"mrr_{eval_k}"],
                    f"val/recall@{eval_k}": val_ranking_metrics[f"recall_{eval_k}"],
                    f"val/map@{eval_k}": val_ranking_metrics[f"map_{eval_k}"],
                    f"val/precision@{eval_k}": val_ranking_metrics[f"precision_{eval_k}"],
                    "lr": lr_now,
                    "recency_gamma": (
                        float(model.recency_gamma.detach().cpu())
                        if getattr(model, "use_recency_bias", False) else None
                    ),
                })

            if verbose:
                print(f"Fold {fold}  Ep {epoch + 1}: "
                     f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, Total loss {total_loss:.3f}, "
                     f"Val NDCG@{eval_k} {val_ranking_metrics[f'ndcg_{eval_k}']:.3f}, "
                     f"Val HR@{eval_k} {val_ranking_metrics[f'hr_{eval_k}']:.3f}, "
                     f"Val MRR@{eval_k} {val_ranking_metrics[f'mrr_{eval_k}']:.3f}, "
                     f"LR {lr_now:.2e}")

            ndcg_now = val_ranking_metrics[f"ndcg_{eval_k}"]
            if ndcg_now > best_ndcg:
                best_ndcg = ndcg_now
                torch.save(model.state_dict(), best_path)
                if track_experiment:
                    run.summary[f"best_val_ndcg@{eval_k}"] = best_ndcg
                print(f"  -> new best, saved -> {best_path}  (NDCG@{eval_k}={best_ndcg:.4f})")

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_ndcg": best_ndcg,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

            # EarlyStopping treats its arg as a loss (lower = better) -> pass -NDCG.
            early_stopping(-ndcg_now, model)
            if early_stopping.early_stop:
                print(f"Early stopping. Best val NDCG@{eval_k}: {best_ndcg:.4f}")
                break

        if track_experiment:
            run.finish()

    print(f"done. best weights per fold saved under {ckpt_dir} as {model_filename.replace('.pth', '_best_f*.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lstm_v2.LSTMAttentionRec using K-fold CV")
    parser.add_argument("-s", "--source", type=str, choices=["dressipi", "trivago", "spotify"],
                        default=None, help="Choose a source dataset.")
    parser.add_argument("-wb", "--wandb", action="store_true", default=False,
                        help="Enable the Weights & Biases experiment tracking.")
    parser.add_argument("-v", "--verbose", action="store_true", default=True,
                        help="Enable verbose output.")
    parser.add_argument("--eval-k", type=int, default=20,
                        help="k used for ranking metrics and early-stopping selection.")
    args = parser.parse_args()

    cfg = load_config(f"config/{args.source}.yaml")
    cfg["source"] = args.source

    train_cv(cfg=cfg, track_experiment=args.wandb, verbose=args.verbose, eval_k=args.eval_k)
