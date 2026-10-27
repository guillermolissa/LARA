"""Convert the Trivago session parquet files into SessionId/ItemId/Time txt files.

Each parquet row is one session with its items already stored, in order, inside the
`sequence_item_ids` list column, alongside a `sequence_timestamps` list column holding
the real per-item unix timestamp (unlike dressipi/spotify, trivago ships actual
per-event times, so no interpolation is needed). For every session we keep only the
last `context_length + 1` items (the model's context window plus the target item) and
explode them into one row per item, preserving the original session order and the
original item order within each session.

ItemId is converted to its integer token id via the project's WordLevel tokenizer
(data/tokenizer-trivago.json); items absent from the vocab map to [UNK]. Trivago's
SessionId is a hashed string (e.g. "76c3f3a6c2a59"), so it is remapped to a sequential
integer id, assigned in order of first appearance across train then test so ids stay
consistent (and disjoint) between the two files. The train split is further divided
into a train/validation pair by sampling whole sessions (20% of sessions go to
validation).

Output files follow the `{source}_train_tr.txt` / `{source}_train_valid.txt` /
`{source}_test.txt` naming convention, written next to each source's parquet files.
"""
import json
from pathlib import Path

import polars as pl
import yaml

SOURCE = "trivago"
CONFIG_PATH = "config/trivago.yaml"
TOKENIZER_PATH = "data/tokenizer-trivago.json"
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
        pl.col("sequence_item_ids").list.tail(seq_len).alias("sequence_item_ids"),
        pl.col("sequence_timestamps").list.tail(seq_len).alias("sequence_timestamps"),
    )

    exploded = df.explode(["sequence_item_ids", "sequence_timestamps"])

    return exploded.select(
        pl.col("session_id").alias("SessionId"),
        pl.col("sequence_item_ids").alias("ItemId"),
        pl.col("sequence_timestamps").cast(pl.Int64).alias("Time"),
    )


def build_session_id_map(*dfs: pl.DataFrame) -> dict:
    """Sequential integer id per unique SessionId, in order of first appearance
    across the given frames (train first, then test)."""
    ordered_ids = pl.concat([df.select("SessionId") for df in dfs]).unique(
        keep="first", maintain_order=True
    ).to_series()
    return {sid: idx for idx, sid in enumerate(ordered_ids.to_list())}


def tokenize_items(df: pl.DataFrame, vocab: dict) -> pl.DataFrame:
    unk_id = vocab["[UNK]"]
    return df.with_columns(
        pl.col("ItemId")
        .cast(pl.String)
        .replace_strict(vocab, default=unk_id, return_dtype=pl.Int64)
        .alias("ItemId")
    )


def integerize_session_ids(df: pl.DataFrame, session_id_map: dict) -> pl.DataFrame:
    return df.with_columns(
        pl.col("SessionId").replace_strict(session_id_map, return_dtype=pl.Int64)
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

    session_id_map = build_session_id_map(df_train, df_test)
    print(f"[{SOURCE}] built sequential session id map for {len(session_id_map)} sessions")

    df_train = integerize_session_ids(df_train, session_id_map)
    df_test = integerize_session_ids(df_test, session_id_map)

    df_train = tokenize_items(df_train, vocab)
    df_test = tokenize_items(df_test, vocab)

    df_train_tr, df_train_valid = split_train_validation(df_train, VALID_FRACTION, seed)

    write_txt(df_train_tr, train_parquet.parent / f"{SOURCE}_train_tr.txt", "train_tr")
    write_txt(df_train_valid, train_parquet.parent / f"{SOURCE}_train_valid.txt", "train_valid")
    write_txt(df_test, test_parquet.parent / f"{SOURCE}_test.txt", "test")


if __name__ == "__main__":
    main()
