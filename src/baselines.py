"""Session-based recommendation baselines.

Baseline models to benchmark a custom sequential model:

    1. MostPopular         - global item frequency, context-agnostic.
    2. FirstOrderMarkov    - P(i_t | i_{t-1}) from the last context item.
    3. AssociationRules    - co-occurrence / confidence w.r.t. the context items.
    4. SessionKNN          - neighbour sessions by set similarity with the context.

Usage
-----
    python src/baselines.py                     # uses the constants below
    python src/baselines.py --n 5 --k 20        # override context / top-k
    python src/baselines.py --models pop markov
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

# --------------------------------------------------------------------------- #
# Configuration (modular - override from the CLI if needed)                    #
# --------------------------------------------------------------------------- #
TRAIN_PATH = "data/train/train_minseql3.parquet"
TEST_PATH = "data/test/test_minseql3.parquet"
OUTPUT_DIR = "submission"

N = 10          # context window: max number of recent items used as input X
K = 20          # number of recommended items returned per session

ITEM_COL = "sequence_item_ids"

# Model-specific hyper-parameters
AR_NORMALIZE = "confidence"   # "count" | "confidence"  (confidence = cooc / freq(candidate))
AR_MAX_SESSION_ITEMS = 50     # cap unique items/session when building co-occurrence
KNN_K_NEIGHBORS = 100         # neighbours kept to score candidates
KNN_SAMPLE_SIZE = 500         # max neighbour sessions inspected per prediction
KNN_INDEX_CAP = 5000          # max (most recent) sessions kept per item in the inverted index
KNN_SIMILARITY = "cosine"     # "cosine" | "jaccard"
SEED = 42


# --------------------------------------------------------------------------- #
# Base class                                                                  #
# --------------------------------------------------------------------------- #
class BaseRecommender:
    """Common interface + Top-K assembly with a MostPopular fallback."""

    name = "base"

    def __init__(self) -> None:
        self.fallback: list[str] = []      # popularity-ordered item ids

    # -- to be implemented by subclasses ----------------------------------- #
    def fit(self, sessions: list[list[str]]) -> "BaseRecommender":
        raise NotImplementedError

    def _rank(self, context: list[str]) -> list[str]:
        """Return candidate items ordered by score (may be shorter than K)."""
        raise NotImplementedError

    # -- shared logic ----------------------------------------------------- #
    def predict(self, context: list[str], k: int = K) -> list[str]:
        ranked = self._rank(context)
        return self._fill(ranked, k, exclude=set(context))

    def predict_batch(self, contexts: list[list[str]], k: int = K) -> list[list[str]]:
        return [self.predict(c, k) for c in contexts]

    def _fill(self, ranked: list[str], k: int, exclude: set[str] = frozenset()) -> list[str]:
        """Deduplicate ``ranked``, then pad with popular items up to ``k``."""
        out: list[str] = []
        seen: set[str] = set()
        for it in ranked:
            if it in seen or it in exclude:
                continue
            seen.add(it)
            out.append(it)
            if len(out) == k:
                return out
        for it in self.fallback:
            if it in seen or it in exclude:
                continue
            seen.add(it)
            out.append(it)
            if len(out) == k:
                break
        return out


# --------------------------------------------------------------------------- #
# 1. Most Popular                                                             #
# --------------------------------------------------------------------------- #
class MostPopular(BaseRecommender):
    name = "pop"

    def fit(self, sessions: list[list[str]]) -> "MostPopular":
        counts: Counter[str] = Counter()
        for s in sessions:
            counts.update(s)
        self.counts = counts
        self.ranked_items = [it for it, _ in counts.most_common()]
        self.fallback = self.ranked_items
        return self

    def _rank(self, context: list[str]) -> list[str]:
        # context-agnostic: the global ranking (fallback padding handles the rest)
        return self.ranked_items


# --------------------------------------------------------------------------- #
# 2. First-Order Markov Chain                                                 #
# --------------------------------------------------------------------------- #
class FirstOrderMarkov(BaseRecommender):
    name = "markov"

    def fit(self, sessions: list[list[str]]) -> "FirstOrderMarkov":
        trans: dict[str, Counter[str]] = defaultdict(Counter)
        for s in sessions:
            for a, b in zip(s, s[1:]):
                trans[a][b] += 1
        # freeze to plain ranked lists for fast inference
        self.transitions = {
            a: [it for it, _ in c.most_common()] for a, c in trans.items()
        }
        return self

    def _rank(self, context: list[str]) -> list[str]:
        if not context:
            return []
        return self.transitions.get(context[-1], [])


# --------------------------------------------------------------------------- #
# 3. Association Rules / Co-occurrence                                        #
# --------------------------------------------------------------------------- #
class AssociationRules(BaseRecommender):
    name = "ar"

    def __init__(self, normalize: str = AR_NORMALIZE,
                 max_session_items: int = AR_MAX_SESSION_ITEMS) -> None:
        super().__init__()
        self.normalize = normalize
        self.max_session_items = max_session_items

    def fit(self, sessions: list[list[str]]) -> "AssociationRules":
        cooc: dict[str, Counter[str]] = defaultdict(Counter)
        freq: Counter[str] = Counter()
        for s in sessions:
            items = list(dict.fromkeys(s))            # unique, keep order
            if len(items) > self.max_session_items:
                items = items[-self.max_session_items:]
            freq.update(items)
            for i, a in enumerate(items):
                for b in items[i + 1:]:
                    cooc[a][b] += 1
                    cooc[b][a] += 1
        self.cooc = cooc
        self.freq = freq
        return self

    def _rank(self, context: list[str]) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        for it in set(context):
            neigh = self.cooc.get(it)
            if not neigh:
                continue
            for cand, c in neigh.items():
                if self.normalize == "confidence":
                    scores[cand] += c / self.freq[cand]
                else:
                    scores[cand] += c
        return [it for it, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# --------------------------------------------------------------------------- #
# 4. Session-KNN                                                              #
# --------------------------------------------------------------------------- #
class SessionKNN(BaseRecommender):
    name = "sknn"

    def __init__(self, k_neighbors: int = KNN_K_NEIGHBORS,
                 sample_size: int = KNN_SAMPLE_SIZE,
                 index_cap: int = KNN_INDEX_CAP,
                 similarity: str = KNN_SIMILARITY,
                 seed: int = SEED) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.sample_size = sample_size
        self.index_cap = index_cap
        self.similarity = similarity
        self.rng = np.random.default_rng(seed)

    def fit(self, sessions: list[list[str]]) -> "SessionKNN":
        self.sessions = [frozenset(s) for s in sessions]
        index: dict[str, list[int]] = defaultdict(list)
        for idx, s in enumerate(self.sessions):
            for it in s:
                index[it].append(idx)
        # Keep only the most recent `index_cap` sessions per item: bounds the
        # per-prediction neighbour gathering for very frequent items.
        self.item_sessions = {
            it: (lst[-self.index_cap:] if len(lst) > self.index_cap else lst)
            for it, lst in index.items()
        }
        return self

    def _rank(self, context: list[str]) -> list[str]:
        ctx = frozenset(context)
        if not ctx:
            return []

        # candidate neighbour sessions = sessions sharing >=1 context item.
        # Cap the pool so very frequent items do not blow up the gather step.
        pool_cap = 4 * self.sample_size
        cand: set[int] = set()
        for it in ctx:
            for idx in self.item_sessions.get(it, ()):
                cand.add(idx)
            if len(cand) >= pool_cap:
                break
        if not cand:
            return []

        cand_arr = np.fromiter(cand, dtype=np.int64, count=len(cand))
        if cand_arr.size > self.sample_size:
            cand_arr = self.rng.choice(cand_arr, size=self.sample_size, replace=False)

        lc = len(ctx)
        sims: list[tuple[float, int]] = []
        for idx in cand_arr:
            s = self.sessions[idx]
            inter = len(ctx & s)
            if not inter:
                continue
            if self.similarity == "jaccard":
                sim = inter / (lc + len(s) - inter)
            else:  # cosine on binary vectors
                sim = inter / math.sqrt(lc * len(s))
            sims.append((sim, int(idx)))

        sims.sort(reverse=True)
        scores: dict[str, float] = defaultdict(float)
        for sim, idx in sims[: self.k_neighbors]:
            for it in self.sessions[idx]:
                if it in ctx:
                    continue
                scores[it] += sim
        return [it for it, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# --------------------------------------------------------------------------- #
# Data helpers                                                                #
# --------------------------------------------------------------------------- #
def load_sessions(path: str) -> list[list[str]]:
    """Full training sequences (stats are fit on the complete session)."""
    return pl.read_parquet(path, columns=[ITEM_COL])[ITEM_COL].to_list()


def load_test(path: str, n: int) -> tuple[list[list[str]], list[list[str]]]:
    """Split each test session into (input context <= n, 1-item target).

    X = sequence[-(n+1):-1]   y = [sequence[-1]]
    """
    df = (
        pl.read_parquet(path, columns=[ITEM_COL])
        .with_columns(_len=pl.col(ITEM_COL).list.len())
        .filter(pl.col("_len") >= 2)
        .select(
            input_items=pl.col(ITEM_COL).list.slice(0, pl.col(ITEM_COL).list.len() - 1).list.tail(n),
            target_items=pl.col(ITEM_COL).list.tail(1),
        )
    )
    return df["input_items"].to_list(), df["target_items"].to_list()


def recall_at_k(preds: list[list[str]], targets: list[list[str]]) -> float:
    hits = sum(1 for p, t in zip(preds, targets) if t[0] in p)
    return hits / len(targets) if targets else 0.0


def mrr_at_k(preds: list[list[str]], targets: list[list[str]]) -> float:
    total = 0.0
    for p, t in zip(preds, targets):
        if t[0] in p:
            total += 1.0 / (p.index(t[0]) + 1)
    return total / len(targets) if targets else 0.0


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {
    "pop": MostPopular,
    "markov": FirstOrderMarkov,
    "ar": AssociationRules,
    "sknn": SessionKNN,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session-based recommendation baselines")
    p.add_argument("--train", default=TRAIN_PATH)
    p.add_argument("--test", default=TEST_PATH)
    p.add_argument("--out", default=OUTPUT_DIR)
    p.add_argument("--n", type=int, default=N, help="context window length")
    p.add_argument("--k", type=int, default=K, help="top-k recommendations")
    p.add_argument("--source", default='dressipi', help="source dataset")
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY),
                   choices=list(MODEL_REGISTRY))
    return p.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading train sessions from {args.train} ...")
    train_sessions = load_sessions(args.train)
    print(f"  {len(train_sessions):,} training sessions")

    print(f"Loading test sessions from {args.test} (N={args.n}) ...")
    inputs, targets = load_test(args.test, args.n)
    print(f"  {len(inputs):,} test sessions")

    # Popularity ranking is shared as the universal fallback.
    pop = MostPopular().fit(train_sessions)

    for key in args.models:
        t0 = time.perf_counter()
        model = pop if key == "pop" else MODEL_REGISTRY[key]().fit(train_sessions)
        model.fallback = pop.ranked_items
        fit_dt = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = model.predict_batch(inputs, args.k)
        pred_dt = time.perf_counter() - t0

        r = recall_at_k(preds, targets)
        m = mrr_at_k(preds, targets)
        print(f"[{key:6s}] recall@{args.k}={r:.4f}  mrr@{args.k}={m:.4f}  "
              f"(fit {fit_dt:.1f}s / infer {pred_dt:.1f}s)")

        pl.DataFrame(
            {"input_items": inputs, "predicted_items": preds, "target_items": targets},
            schema={
                "input_items": pl.List(pl.String),
                "predicted_items": pl.List(pl.String),
                "target_items": pl.List(pl.String),
            },
        ).write_parquet(out_dir / f"{key}_{args.source}_predictions.parquet")
        print(f"         -> {out_dir / f'{key}_{args.source}_predictions.parquet'}")


if __name__ == "__main__":
    main()
