import os
import torch
from dataset import DressipiDataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from custom_collate import test_collate_fn
from utils import set_seed, load_config, load_model, build_model_name
from functools import partial
import model_builder
from pathlib import Path
from tokenizer import get_tokenizer
from tokenizers import Tokenizer
from tqdm import tqdm
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")


SPECIAL_TOKENS = {"[PAD]", "[UNK]", "[SOS]", "[EOS]", "[CLS]", "[SEP]", "[MASK]"}


def predict_topk(model, input_batch: torch.Tensor, top_n: int, use_adaptive_softmax: bool) -> torch.Tensor:
    """Single forward pass — returns top-n item IDs for each sequence in the batch.

    Args:
        model: Trained GPT or GPTAdaSoftmax model.
        input_batch: (B, context_length) token-ID tensor.
        top_n: Number of top items to return per sequence.
        use_adaptive_softmax: Whether the model uses AdaptiveLogSoftmax.

    Returns:
        Tensor of shape (B, top_n) containing the highest-scoring token IDs.
    """
    with torch.no_grad():
        if use_adaptive_softmax:
            scores = model.predict(input_batch)[:, -1, :]  # (B, vocab_size) log-probs
        else:
            scores = model(input_batch)[:, -1, :]          # (B, vocab_size) logits

    return torch.topk(scores, k=top_n, dim=-1).indices    # (B, top_n)


def inference_topk(top_n: int, dynamic_target_length: bool = False):
    print("Init top-k inference")
    print("Loading configuration")
    cfg = load_config("config/config.yaml")
    cfg_model = cfg["model"]
    cfg_data = cfg["data"]
    cfg_hyperparam = cfg["hyperparameters"]
    cfg_experiment = cfg["experiment"]

    requested = cfg_hyperparam.get("device", "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif requested == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    num_workers = cfg_hyperparam["num_workers"] if cfg_hyperparam["num_workers"] is not None else os.cpu_count()

    set_seed(cfg_hyperparam["seed"])

    file_path = Path(cfg_data["test"], cfg_data["test_feature_store"].rsplit(".", 1)[0] + ".parquet")
    tokenizer = get_tokenizer(tokenizer_path=cfg_data["tokenizer_path"])

    print("Loading dataset...")
    test_ds = DressipiDataset(file_path=file_path, tokenizer=tokenizer)
    print(f"Test dataset size: {len(test_ds)}")

    seq_len = cfg_model["context_length"]
    batch_size = cfg_hyperparam["batch_size"]
    cfg_model["items_size"] = tokenizer.get_vocab_size()
    cfg_model["item_meta_embedding"] = cfg_data["item_meta_embedding"]
    cfg_model["item_meta_id_map"] = cfg_data["item_meta_id_map"]

    use_adaptive_softmax = cfg_hyperparam["use_adaptive_softmax"]
    print("Loading model")
    if use_adaptive_softmax:
        print("Using Adaptive Softmax")
        model = model_builder.GPTAdaSoftmaxModel(cfg_model, tokenizer=tokenizer).to(device)
    else:
        print("Using Linear Softmax")
        model = model_builder.GPTModel(cfg_model, tokenizer=tokenizer).to(device)

    load_model(model, target_dir=cfg_model["folder"], model_name=build_model_name(cfg_model, cfg_hyperparam), device=device, weights_only=True)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Clamp top_n to vocab size to avoid topk errors
    vocab_size = tokenizer.get_vocab_size()
    if top_n > vocab_size:
        print(f"Warning: top_n={top_n} exceeds vocab size {vocab_size}; clamping.")
        top_n = vocab_size

    customized_collate_fn = partial(
        test_collate_fn,
        device=device,
        context_length=seq_len,
        pad_token_id=tokenizer.token_to_id("[PAD]"),
        dynamic_target_length=dynamic_target_length,
    )

    test_dataloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    input_items_out = []
    predicted_items_out = []
    target_items_out = []

    # Request extra candidates to account for special tokens that will be filtered
    fetch_n = min(top_n + len(SPECIAL_TOKENS), vocab_size)

    print("Starting top-k inference on test dataset")
    for batch in tqdm(test_dataloader, total=len(test_dataloader), desc="Generating top-k predictions", colour="blue"):
        input_batch, target_batch = batch
        input_batch = input_batch.to(device)

        # Single forward pass for the whole batch
        top_ids = predict_topk(model, input_batch, fetch_n, use_adaptive_softmax)  # (B, fetch_n)

        for i in range(input_batch.size(0)):
            # Decode input context (strip padding)
            input_tokens = [
                tokenizer.id_to_token(tid)
                for tid in input_batch[i].tolist()
                if tokenizer.id_to_token(tid) not in SPECIAL_TOKENS
            ]
            input_items_out.append(input_tokens)

            # Decode top-n predictions, filtering special tokens, up to top_n
            preds = []
            for tid in top_ids[i].tolist():
                token = tokenizer.id_to_token(tid)
                if token not in SPECIAL_TOKENS:
                    preds.append(token)
                if len(preds) == top_n:
                    break
            predicted_items_out.append(preds)

        # Decode targets
        for target_ids in target_batch:
            decoded = [tokenizer.id_to_token(tid) for tid in target_ids.tolist()]
            target_items_out.append(decoded)

    output_json = {
        "input_items": input_items_out,
        "predicted_items": predicted_items_out,
        "target_items": target_items_out,
    }

    df_output = pd.DataFrame(output_json)
    model_stem = build_model_name(cfg_model, cfg_hyperparam).replace(".pth", "")
    dtl_tag = "dyn" if dynamic_target_length else "fixed"
    output_file_path = Path(cfg_data["submission_dir"], f"submit-{model_stem}-{cfg_experiment['run_id']}-{dtl_tag}-topk.parquet")

    print(f"Saving predictions to {output_file_path}")
    os.makedirs(cfg_data["submission_dir"], exist_ok=True)
    df_output.to_parquet(output_file_path, index=False)
    print("Done")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Generate top-N recommendations via a single forward pass (no autoregressive loop)."
    )
    parser.add_argument(
        "-n", "--top_n", type=int, required=True,
        help="Number of top-scoring items to return per sequence. E.g.: 10",
    )
    parser.add_argument(
        "-dtl", "--dynamic_target_length", action="store_true", default=False,
        help="If set, target length is dynamic based on the longest sequence in the batch.",
    )

    args = parser.parse_args()

    inference_topk(
        top_n=args.top_n,
        dynamic_target_length=args.dynamic_target_length,
    )
