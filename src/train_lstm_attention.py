import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from lstm import LSTMAttentionRec
from dataset import ItemDataset
from torch.utils.data import DataLoader , random_split
from custom_collate import collate_fn
from functools import partial
#from data_setup import create_dataloaders
from utils import EarlyStopping, set_seed, load_config, save_model, build_model_name
from torchinfo import summary
from tokenizer import get_tokenizer
from tokenizers import Tokenizer
import torch.multiprocessing as mp
import metrics
warnings.filterwarnings("ignore")  # To ignore user warnings



def load_data(file_path: str,  tokenizer: Tokenizer, validation_ratio: float=0.1):
    

    train_ratio = 1 - validation_ratio


    print("Loading dataset...")
    # It only has the train split, so we divide it overselves
    ds = ItemDataset(file_path=file_path, tokenizer=tokenizer) 
        
    # Keep 90% for training, 10% for validation
    train_ds_size = int(train_ratio * len(ds))
    val_ds_size = len(ds) - train_ds_size
    train_ds, val_ds = random_split(ds, [train_ds_size, val_ds_size])
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    
    return train_ds, val_ds


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.05):
    """Linear warmup then cosine decay to ``min_ratio`` of the base LR."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



def calc_loss_batch(input_batch, target_batch, model, device,
                    use_adaptive_softmax=False, label_smoothing=0.0):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    if use_adaptive_softmax:
        _, loss = model(input_batch, target_batch)
    else:
        logits, _, _ = model(input_batch)
        # Only the last position is used at inference — train against the same target
        logits_last  = logits#[:, -1, :]       # (B, vocab_size)
        targets_last = target_batch#[:, -1]     # (B,)
        loss = torch.nn.functional.cross_entropy(
            logits_last, targets_last,
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None,
                     use_adaptive_softmax=False, label_smoothing=0.0):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                use_adaptive_softmax=use_adaptive_softmax,
                label_smoothing=label_smoothing,
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches



def validate_model(model: torch.nn.Module,
                   train_dataloader: torch.utils.data.DataLoader,
                   val_dataloader: torch.utils.data.DataLoader,
                   device: torch.device,
                   num_batches: int,
                   use_adaptive_softmax: bool = False,
                   label_smoothing: float = 0.0) -> Tuple[float, float]:
    """Evaluates the model on train and val splits."""
    model.eval()
    with torch.no_grad():



        train_loss = calc_loss_loader(
            train_dataloader, model, device, num_batches=num_batches,
            use_adaptive_softmax=use_adaptive_softmax, label_smoothing=label_smoothing,
        )
        val_loss = calc_loss_loader(
            val_dataloader, model, device, num_batches=num_batches,
            use_adaptive_softmax=use_adaptive_softmax, label_smoothing=label_smoothing,
        )
    model.train()
    return train_loss, val_loss


def evaluate_ranking_metrics(model: torch.nn.Module,
                             val_dataloader: torch.utils.data.DataLoader,
                             tokenizer: Tokenizer,
                             device: torch.device,
                             k: int = 10) -> Dict[str, float]:
    """Aggregates validation batch predictions and scores them with metrics.py,
    mirroring the evaluation logic in evaluation.py."""
    model.eval()
    target_items_out, predicted_items_out = [], []

    with torch.no_grad():
        for input_batch, target_batch in val_dataloader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            logits, _, _ = model(input_batch)
            top_ids = torch.topk(logits, k=k, dim=-1).indices  # (B, k)
            last_targets = target_batch#[:, -1]                 # (B,)

            for i in range(input_batch.size(0)):
                target_id = last_targets[i].item()
                if target_id == -100:
                    continue
                predicted_items_out.append(
                    [tokenizer.id_to_token(tid) for tid in top_ids[i].tolist()]
                )
                target_items_out.append([tokenizer.id_to_token(target_id)])

    model.train()

    metric_names = ["hr", "mrr", "precision", "recall", "map", "ndcg"]
    if not target_items_out:
        return {f"{name}_{k}": float("nan") for name in metric_names}

    pairs = list(zip(target_items_out, predicted_items_out))
    hr = np.mean([metrics.hit_rate_k(gt, pred, k=k) for gt, pred in pairs])
    mrr = np.mean([metrics.rr_k(gt, pred, k=k) for gt, pred in pairs])
    precision = np.mean([metrics.precision_k(gt, pred, k=k) for gt, pred in pairs])
    recall = np.mean([metrics.recall_k(gt, pred, k=k) for gt, pred in pairs])
    map_score = np.mean([metrics.apk(gt, pred, k=k) for gt, pred in pairs])
    ndcg = np.mean([metrics.ndcg_k(gt, pred, k=k) for gt, pred in pairs])

    return {
        f"hr_{k}": hr,
        f"mrr_{k}": mrr,
        f"precision_{k}": precision,
        f"recall_{k}": recall,
        f"map_{k}": map_score,
        f"ndcg_{k}": ndcg,
    }

def train_model(cfg: dict, eval_k: int, verbose: bool):

    mp.set_start_method('spawn', force=True)


    cfg_model, cfg_data, cfg_hyperparam = cfg["model"], cfg["data"], cfg["hyperparameters"]

    validation_ratio = cfg_hyperparam["validation_ratio"]
    num_workers = cfg_hyperparam["num_workers"] if cfg_hyperparam["num_workers"] is not None else os.cpu_count()
    
    
    # Setup target device
    requested = cfg_hyperparam.get("device", "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif requested == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if requested != device:
            print("USING DEVICE FOUND: ", device)


    # Set seed for the experiment
    set_seed(cfg_hyperparam["seed"])
    
    
    # load dataset
    file_path = Path(cfg_data["train"], cfg_data["train_feature_store"].rsplit('.', 1)[0] + ".parquet")
    tokenizer_path = cfg_data["tokenizer_path"]
    
    
    # Load tokenizers
    tokenizer = get_tokenizer(tokenizer_path=tokenizer_path)
    PAD_ID = tokenizer.token_to_id("[PAD]")

    # Load dataset
    train_ds, val_ds = load_data(file_path=file_path, tokenizer=tokenizer, validation_ratio=validation_ratio)


    seq_len = cfg_model["context_length"]
    batch_size = cfg_hyperparam["batch_size"]
    cfg_model["items_size"] = tokenizer.get_vocab_size()
    cfg_model["item_meta_embedding"] = cfg_data["item_meta_embedding"]
    cfg_model["item_meta_id_map"] = cfg_data["item_meta_id_map"]

    model_filename = build_model_name(cfg_model, cfg_hyperparam)
    model_filename_checkpoint = build_model_name(cfg_model, cfg_hyperparam).replace(".pth", "_checkpoint.pth")
    
    
    checkpoint_path = Path(cfg_model["folder"], model_filename_checkpoint)

    epochs=cfg_hyperparam["num_epochs"]
    evaluation_frequency=cfg_hyperparam["evaluation_frequency"]
    evaluation_num_batches= None #cfg_hyperparam["evaluation_num_batches"]
    num_accumulation_steps=cfg_hyperparam["num_accumulation_steps"]
    use_adaptive_softmax=cfg_hyperparam["use_adaptive_softmax"]
    warmup_steps=cfg_hyperparam.get("warmup_steps", 0)
    max_grad_norm=cfg_hyperparam.get("max_grad_norm", 1.0)
    label_smoothing=cfg_hyperparam.get("label_smoothing", 0.1)


    customized_collate_fn = partial(
        collate_fn,
        context_length=seq_len,
        pad_token_id=tokenizer.token_to_id("[PAD]"),
    )


    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )


    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )



    if cfg_hyperparam["use_adaptive_softmax"]:
        print("Using Adaptive Softmax")
        #model = model_builder.GPTAdaSoftmaxModel(cfg_model, tokenizer=tokenizer).to(device)
    else:
        print("Using Linear Softmax")
        model = LSTMAttentionRec(
                        embedded_dim=cfg_model["emb_dim"],
                        hidden_dim=cfg_model["hidden_dim"],
                        n_layers=cfg_model["n_layers"],
                        items_size=cfg_model["items_size"],
                        n_heads=cfg_model["n_heads"],
                        drop_rate=cfg_model["drop_rate"],
                        pad_token_id=PAD_ID,
                        tie_weights=cfg_model.get("tie_weights", True),
                        use_recency_bias=cfg_model.get("use_recency_bias", True),
                        recency_decay=cfg_model.get("recency_decay", 0.9),
                        context_length=cfg_model["context_length"]
                    ).to(device)


        # Set loss and optimizer
        #loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = None
    if cfg_hyperparam["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=cfg_hyperparam["learning_rate"], 
                                    weight_decay=cfg_hyperparam["weight_decay"])
    elif cfg_hyperparam["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=cfg_hyperparam["learning_rate"], 
                                    weight_decay=cfg_hyperparam["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer: {cfg_hyperparam['optimizer']}")


    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20
        )
    
    total_steps = epochs * len(train_dataloader)
    scheduler = make_scheduler(optimizer, cfg_hyperparam.get("warmup_steps", 0), total_steps)
    
    label_smoothing = cfg_hyperparam.get("label_smoothing", 0.0)
    max_grad_norm = cfg_hyperparam.get("max_grad_norm", 1.0)
    
    stopper = EarlyStopping(patience=cfg_hyperparam.get("patience", 5), delta=0.0)

    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}")

    model.train()

    train_losses, val_losses, losses = [], [], []
    global_step = -1

    h0, c0 = None, None
    
    best_ndcg = -1.0
    
    print(f"label_smoothing={label_smoothing}  weight_decay={cfg_hyperparam['weight_decay']}  "
            f"lr={cfg_hyperparam['learning_rate']}  warmup={cfg_hyperparam.get('warmup_steps', 0)}  "
            f"total_steps={total_steps}  early-stop on val NDCG@{eval_k}")
    
    
    
    for epoch in range(start_epoch, epochs):

        total_loss = 0
        running = 0.0

        for input_batch, target_batch in tqdm(train_dataloader, desc=f"epoch {epoch}", colour="green"):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            #target_batch = target_batch[:, -1].unsqueeze(1)
        
            
            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                use_adaptive_softmax=use_adaptive_softmax,
                label_smoothing=label_smoothing,
            )


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss = total_loss + loss.item()
        


        train_loss, val_loss = validate_model(
            model, train_dataloader, val_dataloader, device,
            num_batches=evaluation_num_batches,
            use_adaptive_softmax=use_adaptive_softmax,
            label_smoothing=label_smoothing,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        losses.append(total_loss)

                #if run is not None:
                #    run.log({"train/train_loss": train_loss, "val/val_loss": val_loss})


        val_ranking_metrics = evaluate_ranking_metrics(
                        model, val_dataloader, tokenizer, device, k=eval_k,
                    )
        
        if verbose:    
            print(f"Ep {epoch+1} (Step {global_step}): "
                f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, Total loss {total_loss:.3f}, "
                f"Val HR@{eval_k} {val_ranking_metrics[f'hr_{eval_k}']:.3f}, Val MRR@{eval_k} {val_ranking_metrics[f'mrr_{eval_k}']:.3f}, "
                f"Val Precision@{eval_k} {val_ranking_metrics[f'precision_{eval_k}']:.3f}, Val Recall@{eval_k} {val_ranking_metrics[f'recall_{eval_k}']:.3f}, "
                f"Val MAP@{eval_k} {val_ranking_metrics[f'map_{eval_k}']:.3f}, Val NDCG@{eval_k} {val_ranking_metrics[f'ndcg_{eval_k}']:.3f}, "
                f"LR {scheduler.get_last_lr()[0]:.2e}")

        
        
        # EarlyStopping treats its arg as a loss (lower = better) → pass -NDCG.
        stopper(-val_ranking_metrics[f"ndcg_{eval_k}"], model)
        if val_ranking_metrics[f"ndcg_{eval_k}"] > best_ndcg:
            best_ndcg = val_ranking_metrics[f"ndcg_{eval_k}"]

            checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                    }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"  ↳ new best, saved → {checkpoint_path}  (NDCG@{eval_k}={best_ndcg:.4f})")

        if stopper.early_stop:
            print(f"early stopping at epoch {epoch}")
            break


    print(f"done. best val NDCG@{args.eval_k}={best_ndcg:.4f}  weights → {checkpoint_path}")
    # Save the model with help from utils.py
    save_model(model=model,
                    target_dir=cfg_model["folder"],
                    model_name=model_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a LARA model using CV')
    parser.add_argument('-s', '--source', type=str, default='dressipi',
                        help='Choose a source. Available options are `dressipi`, `trivago` and `spotify`')
    
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output.')
    
    
    parser.add_argument("--eval-k", type=int, default=20)



    args = parser.parse_args()

    #assert(args.source not in ['dressipi', 'trivago', 'spotify'], "Available options for source are `dressipi`, `trivago` and `spotify`")

    cfg =None
    if args.source == 'dressipi':
        cfg = load_config("config/dressipi.yaml")
    elif args.source == 'trivago':
        cfg = load_config("config/trivago.yaml")
    else: 
        cfg = load_config("config/spotify.yaml")

    cfg['source']=args.source


    train_model(cfg=cfg, eval_k=args.eval_k, verbose=args.verbose)