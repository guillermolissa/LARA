import os
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from utils import load_config, build_model_name
import argparse
import time


# Import metrics from metrics.py
import metrics


def get_metric_functions(module):
    # Get all callables in metrics.py that don't start with "_"
    return [getattr(module, name) for name in dir(module)
            if callable(getattr(module, name)) and not name.startswith("_")]

def evaluate(input_file: str = None, dynamic_target_length: bool = False):

     # Load configuration
    print("Loading configuration")
    # Load config for experiment
    cfg = load_config("config/config.yaml")
    cfg_model = cfg["model"]
    cfg_data = cfg["data"]
    cfg_hyperparam = cfg["hyperparameters"]
    cfg_experiment = cfg["experiment"]

    # Resolve input file: use provided name or default to the inference output filename
    if input_file is None:
        model_stem = build_model_name(cfg_model, cfg_hyperparam).replace(".pth", "")
        dtl_tag = "dyn" if dynamic_target_length else "fixed"
        resolved_name = f"submit-{model_stem}-{cfg_experiment['run_id']}-{dtl_tag}.parquet"
        print(f"No file specified — using inference output: {resolved_name}")
    else:
        resolved_name = f"{input_file.rsplit('.', 1)[0]}.parquet"

    print("Loading data")
    # Load parquet file
    input_file_path = Path(cfg_data["submission_dir"], resolved_name)
    output_file_path = Path(cfg_data["submission_dir"], "metrics.csv")

    if not input_file_path.exists():
        raise FileNotFoundError(f"Submission file not found: {input_file_path}")

    df = pd.read_parquet(input_file_path)

    # Get all metric functions
    metric_functions = get_metric_functions(metrics)

    print(f"Found {len(metric_functions)} metric functions in metrics.py")
    # Apply each metric and store results
    results = {"file_name": resolved_name, "datetime": time.strftime("%Y-%m-%d %H:%M:%S")}
    
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
            


    # Save results as a single-row csv file
    results_df = pd.DataFrame([results])
    print(results)
    if Path(output_file_path).exists():
        # If the file exists, append without header
        results_df.to_csv(output_file_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(output_file_path, index=False)

    print(f"Saved metrics to {output_file_path}")
    print("Evaluation completed successfully")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model predictions.')
    parser.add_argument('-f', '--file_name', type=str, default=None,
                        help='Submission file to evaluate (parquet). Defaults to the inference output file derived from config.')
    parser.add_argument('-dtl', '--dynamic_target_length', action='store_true', default=False,
                        help='Must match the flag used during inference to resolve the correct default filename.')

    args = parser.parse_args()

    evaluate(args.file_name, dynamic_target_length=args.dynamic_target_length)