"""Convert the Dressipi session parquet files into SessionId/ItemId/Time txt files.

Each parquet row is one session with its items already stored, in order, inside the
`sequence_item_ids` list column. For every session we keep only the last
`context_length + 1` items (the model's context window plus the target item) and
explode them into one row per item, preserving the original session order and the
original item order within each session.

Per-item timestamps don't exist in the source data (only session_start_date /
session_end_date are available), so `Time` is linearly interpolated between those two
bounds according to each item's position in the (possibly truncated) sequence. This
keeps Time strictly increasing within a session, matching the sequence order, without
inventing precision the source data doesn't have.

ItemId is converted to its integer token id via the project's WordLevel tokenizer
(data/tokenizer-dressipi.json); items absent from the vocab map to [UNK]. The train
split is further divided into a train/validation pair by sampling whole sessions
(20% of sessions go to validation), as dressipi SessionId is already numeric.

Output files follow the `{source}_train_tr.txt` / `{source}_train_valid.txt` /
`{source}_test.txt` naming convention, written next to each source's parquet files.
"""
import json
from pathlib import Path

import polars as pl
import yaml

SOURCE = "dressipi"
CONFIG_PATH = "config/dressipi.yaml"
TOKENIZER_PATH = "data/tokenizer-dressipi.json"
VALID_FRACTION = 0.2


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_vocab(tokenizer_path: str) -> dict:
    with open(tokenizer_path, "r") as f:
        data = json.load(f)
    return data["model"]["vocab"]


def build_session_rows(df: pl.DataFrame, seq_len: int) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("sequence_item_ids").list.tail(seq_len).alias("sequence_item_ids")
    ).with_columns(pl.col("sequence_item_ids").list.len().alias("trunc_len"))

    df = df.with_columns(pl.int_ranges(0, pl.col("trunc_len")).alias("position"))

    exploded = df.explode(["sequence_item_ids", "position"])

    start_epoch = pl.col("session_start_date").dt.epoch(time_unit="s")
    end_epoch = pl.col("session_end_date").dt.epoch(time_unit="s")

    exploded = exploded.with_columns(
        pl.when(pl.col("trunc_len") > 1)
        .then(
            start_epoch
            + (pl.col("position") / (pl.col("trunc_len") - 1))
            * (end_epoch - start_epoch)
        )
        .otherwise(start_epoch)
        .alias("Time")
    )

    return exploded.select(
        pl.col("session_id").cast(pl.Int64).alias("SessionId"),
        pl.col("sequence_item_ids").alias("ItemId"),
        pl.col("Time").cast(pl.Int64),
    )


def tokenize_items(df: pl.DataFrame, vocab: dict) -> pl.DataFrame:
    unk_id = vocab["[UNK]"]
    return df.with_columns(
        pl.col("ItemId")
        .cast(pl.String)
        .replace_strict(vocab, default=unk_id, return_dtype=pl.Int64)
        .alias("ItemId")
    )


def split_train_validation(df: pl.DataFrame, valid_fraction: float, seed: int):
    session_ids = df.select("SessionId").unique().to_series()
    valid_ids = session_ids.sample(
        fraction=valid_fraction, with_replacement=False, seed=seed
    )
    valid_mask = pl.col("SessionId").is_in(valid_ids)
    df_valid = df.filter(valid_mask)
    df_tr = df.filter(~valid_mask)
    return df_tr, df_valid


def write_txt(df: pl.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, include_header=True, separator="\t")
    n_sessions = df.select("SessionId").n_unique()
    print(f"[{SOURCE}] {label} -> {path} ({df.height} rows, {n_sessions} sessions)")


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    context_length = cfg["model"]["context_length"]
    seed = cfg["hyperparameters"]["seed"]
    seq_len = context_length + 1
    print(f"[{SOURCE}] context_length={context_length} -> keeping last {seq_len} items per session")

    vocab = load_vocab(TOKENIZER_PATH)
    print(f"[{SOURCE}] loaded tokenizer vocab from {TOKENIZER_PATH} ({len(vocab)} tokens)")

    train_parquet = Path(f"data/train/{SOURCE}/train-{SOURCE}.parquet")
    test_parquet = Path(f"data/test/{SOURCE}/test-{SOURCE}.parquet")

    df_train = build_session_rows(pl.read_parquet(train_parquet), seq_len)
    df_test = build_session_rows(pl.read_parquet(test_parquet), seq_len)

    df_train = tokenize_items(df_train, vocab)
    df_test = tokenize_items(df_test, vocab)

    df_train_tr, df_train_valid = split_train_validation(df_train, VALID_FRACTION, seed)

    write_txt(df_train_tr, train_parquet.parent / f"{SOURCE}_train_tr.txt", "train_tr")
    write_txt(df_train_valid, train_parquet.parent / f"{SOURCE}_train_valid.txt", "train_valid")
    write_txt(df_test, test_parquet.parent / f"{SOURCE}_test.txt", "test")


if __name__ == "__main__":
    main()
