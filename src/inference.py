import os
import torch
from dataset import DressipiDataset
from torch.utils.data import DataLoader 
import torch.multiprocessing as mp
from custom_collate import test_collate_fn
from utils import set_seed, load_config, load_model, build_model_name
from functools import partial
import model_builder   # Adjust the import based on the actual location of GPTModel
from torchinfo import summary
from pathlib import Path
from tokenizer import get_tokenizer
from tokenizers import Tokenizer
from tqdm import tqdm
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")


def load_data(file_path: str,  tokenizer: Tokenizer):
    

    print("Loading dataset...")
    # It only has the train split, so we divide it overselves
    test_ds = DressipiDataset(file_path=file_path, tokenizer=tokenizer) 
        
    print(f"Test dataset size: {len(test_ds)}")

    return test_ds


def generate_simple_items(model, idx, max_new_items, context_size, use_adaptive_softmax=False):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_items):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            if use_adaptive_softmax:
                logits = model.predict(idx_cond)
                logits = logits[:, -1, :]  # (B, vocab_size) — last time step, all batch items
            else:
                logits = model(idx_cond)
                # Focus only on the last time step
                # (batch, n_token, vocab_size) becomes (batch, vocab_size)
                logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx



def inference(max_new_tokens: int, dynamic_target_length: bool = False, model_path: str = None):
    print("Init inference")
    print("Loading configuration")
    # Load config for prediction
    cfg_model = load_config("config/config.yaml")["model"]
    cfg_data = load_config("config/config.yaml")["data"]
    cfg_hyperparam = load_config("config/config.yaml")["hyperparameters"]
    cfg_experiment = load_config("config/config.yaml")["experiment"]

    # Setup target device
    # Setup target device
    requested = cfg_hyperparam.get("device", "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif requested == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"


    num_workers = cfg_hyperparam["num_workers"] if cfg_hyperparam["num_workers"] is not None else os.cpu_count()
   

    # Set seed for the experiment
    set_seed(cfg_hyperparam['seed'])

     # load dataset
    file_path = Path(cfg_data["test"], cfg_data["test_feature_store"].rsplit('.', 1)[0] + ".parquet")
    tokenizer_path = cfg_data["tokenizer_path"]
    
    
    # Load tokenizers
    tokenizer = get_tokenizer(tokenizer_path=tokenizer_path)
    
    # Load dataset
    test_ds = load_data(file_path=file_path, tokenizer=tokenizer)
    
    seq_len = cfg_model["context_length"]
    batch_size = cfg_hyperparam["batch_size"]
    cfg_model["items_size"] = tokenizer.get_vocab_size()
    cfg_model["item_meta_embedding"] = cfg_data["item_meta_embedding"]
    cfg_model["item_meta_id_map"] = cfg_data["item_meta_id_map"]

    print("Loading model")   
    # Create model with help from model_builder.py
    
    if cfg_hyperparam["use_adaptive_softmax"]:
        print("Using Adaptive Softmax")
        model = model_builder.GPTAdaSoftmaxModel(cfg_model, tokenizer=tokenizer).to(device)
    else:
        print("Using Linear Softmax")
        model = model_builder.GPTModel(cfg_model, tokenizer=tokenizer).to(device)

    # Load model weights — use explicit path if provided, otherwise derive from config
    if model_path is not None:
        model_path = Path(model_path)
        load_model(model, target_dir=str(model_path.parent), model_name=model_path.name, device=device)
    else:
        load_model(model, target_dir=cfg_model['folder'], model_name=build_model_name(cfg_model, cfg_hyperparam), device=device)
    model.eval()


    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Create test DataLoader with custom collate function
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
        num_workers=num_workers
    )


    # # Load test dataset
    # test_dataloader = create_test_dataloader(
    #     file_path=Path(cfg_data["test"], cfg_data["feature_store_name"].rsplit('.', 1)[0] + ".parquet"),
    #     tokenizer_path=cfg_data["tokenizer_path"],
    #     seq_len=cfg_model["context_length"],
    #     n_first=n_first,
    #     batch_size=cfg_hyperparam["batch_size"]
    # )

    # Initialize list to store output predictions
    output_pred_items = []
    input_items = []
    #target_ids_list = []
    target_items_list = []

    print("Starting inference on test dataset")
    # Iterate over the test dataset
    for batch in tqdm(test_dataloader, total=len(test_dataloader), desc="Generating predictions", colour="blue"):
        input_batch_ids, target_batch_ids = batch

        for input_ids in input_batch_ids:
            encoded_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)  # Add batch dimension and move to device
            
            model_output = generate_simple_items(
                model=model,
                idx=encoded_tensor,
                max_new_items=max_new_tokens,
                context_size=cfg_model["context_length"],
                use_adaptive_softmax=cfg_hyperparam["use_adaptive_softmax"]
            )
            
            out_to_decode = model_output.squeeze(0).tolist()
            
            decoded_items_pred = list(map(tokenizer.id_to_token, out_to_decode))
            
            input_items.append(decoded_items_pred[:seq_len])
            output_pred_items.append(decoded_items_pred[seq_len:])
            

        for target_ids in target_batch_ids:            
            #target_ids_list.append(target_ids.numpy().tolist())
            decoded_items_target = list(map(tokenizer.id_to_token, target_ids))
            target_items_list.append(decoded_items_target)
            
            
    # Save the predictions and targets to a text file
    output_json = {
        "input_items": input_items,
        "predicted_items": output_pred_items,
        "target_items": target_items_list
    }

    df_output = pd.DataFrame(output_json)
    model_stem = build_model_name(cfg_model, cfg_hyperparam).replace(".pth", "")
    dtl_tag = "dyn" if dynamic_target_length else "fixed"
    output_file_path = Path(cfg_data["submission_dir"], f"submit-{model_stem}-{cfg_experiment['run_id']}-{dtl_tag}.parquet")

    print(f"Saving predictions to {output_file_path}")
    if not os.path.exists(cfg_data["submission_dir"]):
        os.makedirs(cfg_data["submission_dir"])
    df_output.to_parquet(output_file_path, index=False)


if __name__ == "__main__":
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description='Load test data and generate predictions using a trained model.')
    parser.add_argument('-mnt', '--max_new_items', type=int, required=True, help='Number of new items to generate. Ej: 8')
    parser.add_argument('-dtl', '--dynamic_target_length', action='store_true', default=False, help='If set, target length is dynamic based on the longest sequence in the batch.')
    parser.add_argument('-mp', '--model_path', type=str, default=None, help='Path to a model .pth file to load for inference. Overrides the model derived from config. Ej: models/my_model.pth')

    args = parser.parse_args()

    inference(
        max_new_tokens=args.max_new_items,
        dynamic_target_length=args.dynamic_target_length,
        model_path=args.model_path,
    )
