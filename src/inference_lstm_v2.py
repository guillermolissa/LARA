"""
inference_lstm_v2.py — top-K submission generator for ``lstm_v2.LSTMAttentionRec``.

Mirrors ``inference_lara_topk.py`` but uses the v2 model / collate and reads
the logits at the last *real* step (``model.predict_last``) instead of a fixed
``[:, -1]`` slice.

Output schema matches the baselines / existing submissions:
    input_items      List(String)   context items actually fed to the model
    predicted_items  List(String)   top-K item ids, score-ordered
    target_items     List(String)   1-element ground-truth list

Usage:
    python src/inference_lstm_v2.py --source dressipi --top_n 20
"""

from __future__ import annotations

import argparse
import os
import warnings
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_collate import IGNORE_INDEX, collate_train_fn
from dataset import ItemDataset
from lstm_v2 import LSTMAttentionRec
from tokenizer import get_tokenizer
from utils import build_model_name, load_config, load_model, set_seed

warnings.filterwarnings("ignore")

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[CLS]", "[SEP]", "[MASK]"]


def pick_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Top-K inference for lstm_v2")
    ap.add_argument("-s", "--source", choices=["dressipi", "trivago", "spotify"], required=True)
    ap.add_argument("-n", "--top_n", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(f"config/{args.source}.yaml")
    m, d, hp, ex = cfg["model"], cfg["data"], cfg["hyperparameters"], cfg["experiment"]

    set_seed(hp["seed"])
    device = pick_device(hp.get("device", "cpu"))
    print(f"device={device}")

    tokenizer = get_tokenizer(tokenizer_path=d["tokenizer_path"])
    pad_id = tokenizer.token_to_id("[PAD]")
    special_ids = [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS
                   if tokenizer.token_to_id(t) is not None]
    m["items_size"] = tokenizer.get_vocab_size()

    test_file = Path(d["test"], d["test_feature_store"].rsplit(".", 1)[0] + ".parquet")
    test_ds = ItemDataset(file_path=test_file, tokenizer=tokenizer)
    print(f"test dataset={len(test_ds):,}")

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
    model_name = build_model_name(m, hp).replace(".pth", "_v2.pth")
    load_model(model, target_dir=m["folder"], model_name=model_name,
               device=device, weights_only=True)
    model.eval()

    top_n = min(args.top_n, tokenizer.get_vocab_size())
    collate = partial(collate_train_fn, pad_token_id=pad_id, context_length=m["context_length"])
    loader = DataLoader(test_ds, batch_size=hp["batch_size"], shuffle=False,
                        drop_last=False, collate_fn=collate,
                        num_workers=hp.get("num_workers") or 0)

    input_items, predicted_items, target_items = [], [], []

    for inputs, targets, lengths in tqdm(loader, desc="inference", colour="blue"):
        inputs, lengths = inputs.to(device), lengths.to(device)
        last_logits = model.predict_last(inputs, lengths, mask_token_ids=special_ids)
        top_ids = torch.topk(last_logits, k=top_n, dim=-1).indices.tolist()
        tgt_pos = (lengths - 1).tolist()

        for i, row in enumerate(top_ids):
            n = int(lengths[i].item())
            ctx = [tokenizer.id_to_token(t) for t in inputs[i, :n].tolist()]
            input_items.append([c for c in ctx if c not in SPECIAL_TOKENS])
            predicted_items.append([tokenizer.id_to_token(t) for t in row])
            gt_id = int(targets[i, tgt_pos[i]].item())
            target_items.append([tokenizer.id_to_token(gt_id)] if gt_id != IGNORE_INDEX else [])

    out = pd.DataFrame(
        {"input_items": input_items, "predicted_items": predicted_items, "target_items": target_items}
    )
    out_dir = d["submission_dir"]
    os.makedirs(out_dir, exist_ok=True)
    stem = model_name.replace(".pth", "")
    out_path = Path(out_dir, f"submit-{stem}-{ex.get('run_id', 'run')}-topk.parquet")
    out.to_parquet(out_path, index=False)
    print(f"saved → {out_path}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
