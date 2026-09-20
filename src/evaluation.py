import os
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from utils import load_config, build_model_name
import argparse
import time


# Import metrics from metrics.py
import metrics

# Known dataset sources — also the submission subdirectory names written by the
# inference scripts (see src/inference_lara_topk*.py), e.g. submission/dressipi/*.parquet
SOURCES = ["dressipi", "trivago", "spotify"]


def get_metric_functions(module):
    # Get all callables in metrics.py that don't start with "_"
    return [getattr(module, name) for name in dir(module)
            if callable(getattr(module, name)) and not name.startswith("_")]


def infer_source(path: Path) -> str:
    """Best-effort source for a single, explicitly-passed file (-f/--file_path),
    which may not live under a submission/<source>/ subdirectory. Tries the
    parent directory name first, then falls back to a substring match on the
    filename itself (submission files are typically named like
    'submit-dressipi-...' or 'dressipi_iknn_predictions.parquet')."""
    if path.parent.name in SOURCES:
        return path.parent.name
    stem = path.stem.lower()
    for source in SOURCES:
        if source in stem:
            return source
    return "unknown"


def discover_submission_files(submission_dir: Path) -> list[tuple[Path, str]]:
    """Default file discovery: submission_dir/<source>/*.parquet for each known source."""
    files = []
    for source in SOURCES:
        source_dir = submission_dir / source
        matches = sorted(source_dir.glob("*.parquet"))
        if not matches:
            print(f"  (no .parquet files found in {source_dir})")
            continue
        files.extend((match, source) for match in matches)
    return files


def _normalize_string_view_field(field: pa.Field) -> pa.Field:
    """Rewrite a `string_view` (or `list<string_view>`) field to plain `string`.

    Some writers (e.g. Polars/DuckDB) emit Arrow's newer `string_view` type for
    list-of-string columns. This pyarrow/pandas version's `to_pandas()` doesn't
    support converting that type when nested in a list
    (``ArrowNotImplementedError: Not implemented type for Arrow list to
    pandas: string_view``), even though reading the raw Arrow table works
    fine — so normalise it before converting to pandas.
    """
    t = field.type
    if pa.types.is_string_view(t):
        return field.with_type(pa.string())
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        if pa.types.is_string_view(t.value_type):
            list_ctor = pa.large_list if pa.types.is_large_list(t) else pa.list_
            return field.with_type(list_ctor(pa.string()))
    return field


def load_parquet(path: Path) -> pd.DataFrame:
    """Read a submission parquet file regardless of which tool wrote it."""
    table = pq.read_table(path)
    new_schema = pa.schema([_normalize_string_view_field(f) for f in table.schema])
    if new_schema != table.schema:
        table = table.cast(new_schema)
    return table.to_pandas()


def evaluate_file(input_file_path: Path, source: str, output_file_path: Path) -> None:
    print(f"Loading data from {input_file_path}")

    if not input_file_path.exists():
        raise FileNotFoundError(f"Submission file not found: {input_file_path}")

    df = load_parquet(input_file_path)

    # Get all metric functions
    metric_functions = get_metric_functions(metrics)

    print(f"Found {len(metric_functions)} metric functions in metrics.py")
    # Apply each metric and store results
    file_name = input_file_path.stem
    results = {"file_name": file_name, "datetime": time.strftime("%Y-%m-%d %H:%M:%S")}

    for funk in metric_functions:
        try:
            if funk.__name__ in ["hit_rate_k", "precision_k", "recall_k", "apk", "ndcg_k", "rr_k"]:
                print(f"Applying metric: {funk.__name__} with k parameter")
                # If the function requires a parameter, pass it
                for k in [1, 3, 5, 10, 20]:
                    #funk = func(k=k)
                    df[f"{funk.__name__}_{k}"] = df.apply(lambda x: funk(list(x["target_items"]), list(x["predicted_items"]), k=k), axis=1)

                    # Calculate mean of all numeric metrics
                    if funk.__name__ == "apk":
                        results[f"map@{k}".upper()] = df[f"{funk.__name__}_{k}"].mean()
                    elif funk.__name__ == "rr_k":
                        results[f"mrr@{k}".upper()] = df[f"{funk.__name__}_{k}"].mean()
                    else:
                        results[f"{funk.__name__}{k}".replace("k", "@").upper().replace("_", "")] = df[f"{funk.__name__}_{k}"].mean()
            else:
                    continue
        except Exception as e:
            print(f"Error applying {funk.__name__}: {e}")
            #º.error(f"Error applying {func.__name__}: {e}")

    # Source tracking column: which dataset (spotify/dressipi/trivago) this
    # submission file was generated from.
    results["source"] = source

    # Save results as a single-row csv file
    results_df = pd.DataFrame([results])
    print(results)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    if output_file_path.exists():
        # If the file exists, append without header
        results_df.to_csv(output_file_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(output_file_path, index=False)

    print(f"Saved metrics to {output_file_path}")


def evaluate(input_file_path: str = None):

     # Load configuration
    print("Loading configuration")
    # Load config for experiment
    #cfg = load_config("config/config.yaml")
    #cfg_data = cfg["data"]

    output_file_path = Path("outputs", "metrics.csv")

    if input_file_path is not None:
        # Single-file mode: evaluate the file passed via -f/--file_path.
        file_path = Path(input_file_path)
        targets = [(file_path, infer_source(file_path))]
    else:
        # Evaluate every .parquet file under submission/<source>/*.parquet
        # for each known source (dressipi, trivago, spotify).
        submission_dir = Path("submission")#Path(cfg_data["submission_dir"])
        print(f"No file specified — scanning {submission_dir}/<source>/*.parquet "
              f"for source in {SOURCES}")
        targets = discover_submission_files(submission_dir)
        if not targets:
            raise FileNotFoundError(
                f"No .parquet files found under {submission_dir}/<source>/*.parquet "
                f"for source in {SOURCES}"
            )

    for file_path, source in targets:
        evaluate_file(file_path, source, output_file_path)

    print("Evaluation completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model predictions.')
    parser.add_argument('-f', '--file_path', type=str, default=None,
                        help='Submission file to evaluate (parquet). If omitted, evaluates every '
                             '.parquet file under submission/<source>/*.parquet for dressipi, trivago and spotify.')

    args = parser.parse_args()

    evaluate(args.file_path)
