import math
import os
import warnings
from pathlib import Path
import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from lstm import LSTMAttentionModel
from dataset import DressipiDataset
from torch.utils.data import DataLoader , random_split
from custom_collate import collate_fn
from functools import partial
#from data_setup import create_dataloaders
from utils import set_seed, load_config, save_model, build_model_name, EarlyStopping
from torchinfo import summary
from tokenizer import get_tokenizer
from tokenizers import Tokenizer
import torch.multiprocessing as mp
import wandb
import metrics
warnings.filterwarnings("ignore")  # To ignore user warnings
from sklearn.model_selection import KFold



def load_data(file_path: str,  tokenizer: Tokenizer):
    
    print("Loading dataset...")
    # It only has the train split, so we divide it overselves
    ds = DressipiDataset(file_path=file_path, tokenizer=tokenizer) 
        
    return ds

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


# def _warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
#     def lr_lambda(current_step: int):
#         if current_step < warmup_steps:
#             return float(current_step) / float(max(1, warmup_steps))
#         progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
#         return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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


#train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
#val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

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

if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)
    


    cfg = load_config("config/config.yaml")
    cfg_model, cfg_data, cfg_hyperparam, cfg_experiment = cfg["model"], cfg["data"], cfg["hyperparameters"], cfg["experiment"]

    # Set WandB API key
    os.environ['WANDB_API_KEY'] = cfg_experiment["wandb_api_key"]

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


    # Set seed for the experiment
    set_seed(cfg_hyperparam["seed"])
    k_folds = cfg_hyperparam["kfold"]
    
    
    # load dataset
    file_path = Path(cfg_data["train"], cfg_data["train_feature_store"].rsplit('.', 1)[0] + ".parquet")
    tokenizer_path = cfg_data["tokenizer_path"]
    
    
    # Load tokenizers
    tokenizer = get_tokenizer(tokenizer_path=tokenizer_path)

    seq_len = cfg_model["context_length"]
    batch_size = cfg_hyperparam["batch_size"]
    cfg_model["items_size"] = tokenizer.get_vocab_size()
    cfg_model["item_meta_embedding"] = cfg_data["item_meta_embedding"]
    cfg_model["item_meta_id_map"] = cfg_data["item_meta_id_map"]

        
    

    
    model_filename = build_model_name(cfg_model, cfg_hyperparam)
    


    

    # Load dataset
    dataset = load_data(file_path=file_path, tokenizer=tokenizer)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    


    epochs=cfg_hyperparam["num_epochs"]
    evaluation_frequency=cfg_hyperparam["evaluation_frequency"]
    evaluation_num_batches= None #cfg_hyperparam["evaluation_num_batches"]
    num_accumulation_steps=cfg_hyperparam["num_accumulation_steps"]
    use_adaptive_softmax=cfg_hyperparam["use_adaptive_softmax"]
    warmup_steps=cfg_hyperparam.get("warmup_steps", 0)
    max_grad_norm=cfg_hyperparam.get("max_grad_norm", 1.0)
    label_smoothing=cfg_hyperparam.get("label_smoothing", 0.1)


    GROUP = cfg_experiment["group"] + wandb.util.generate_id()

    train_total_loss, val_total_loss = [], []
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

        # Print
        print(f'FOLD {fold}')
        print('--------------------------------'*2)


        if cfg_hyperparam["use_adaptive_softmax"]:
                    print("Using Adaptive Softmax")
                    #model = model_builder.GPTAdaSoftmaxModel(cfg_model, tokenizer=tokenizer).to(device)
        else:
            print("Using Linear Softmax")
            model = LSTMAttentionModel(
                embedded_dim=cfg_model["emb_dim"],
                hidden_dim=cfg_model["hidden_dim"],
                layer_dim=cfg_model["n_layers"],
                items_size=cfg_model["items_size"],
                n_head=cfg_model["n_heads"],
                drop_rate=cfg_model["drop_rate"],
                context_length=cfg_model["context_length"]
            ).to(device)
    
        
        
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


        # config file to save in wandb
        config = cfg_model | cfg_hyperparam
        config["train_feature_store"]= cfg_data["train_feature_store"]
    
        if cfg_hyperparam["use_adaptive_softmax"]:
            config["cutoffs"] = cfg_model["cutoffs"]
            config["div_value"] = cfg_model["div_value"]
            config["loss_fn"] = "AdaptiveSoftmaxLoss"
        else:
            config["loss_fn"] = "CrossEntropyLoss"
            config["cutoffs"] = None
            config["div_value"] = None
    
        config["model_name"] = model_filename 

        config["fold"] = fold

        run_name = f"fold-{fold}"

        run = wandb.init(entity=cfg_experiment["entity"], project=cfg_experiment["project"]
                         #,id=cfg_experiment["run_id"]
                         ,resume=cfg_experiment["resume"]
                         ,group=GROUP
                         , name=run_name
                         ,job_type=run_name
                         ,reinit=True, config=config)


            
        print("RUN WANDB INFO\n")
        print("ENTITY: ", cfg_experiment["entity"], " - PROJECT: ", cfg_experiment["project"] ," - GROUP: ", GROUP)

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)


        

        checkpoint_path = Path(cfg_model["folder"]
                            , model_filename.replace(".pth", f"_checkpoint_f{fold}.pth"))


        early_stopping = EarlyStopping(patience=5, delta=0.01)

        customized_collate_fn = partial(
            collate_fn,
            context_length=seq_len,
            pad_token_id=tokenizer.token_to_id("[PAD]"),
        )


        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_subsampler,
            collate_fn=customized_collate_fn,
            drop_last=True,
            num_workers=num_workers
        )


        val_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_subsampler,
            collate_fn=customized_collate_fn,
            drop_last=False,
            num_workers=num_workers
        )


        start_epoch = 0
        best_loss = float('inf')

        model.apply(reset_weights)
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
        
        for epoch in tqdm(range(start_epoch, epochs), total=epochs, desc="Training...", colour="orange"):

            total_loss = 0

            for batch_idx, (input_batch, target_batch) in enumerate(train_dataloader):

                #target_batch = target_batch[:, -1].unsqueeze(1)
            
                
                optimizer.zero_grad()

                loss = calc_loss_batch(
                    input_batch, target_batch, model, device,
                    use_adaptive_softmax=use_adaptive_softmax,
                    label_smoothing=label_smoothing,
                )


                loss.backward()


                    
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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

            val_ranking_metrics = evaluate_ranking_metrics(
                model, val_dataloader, tokenizer, device, k=10,
            )

            run.log({
                "epoch": (epoch+1),
                "train/train_loss": train_loss,
                "val/loss": val_loss,
                "val/hr_10": val_ranking_metrics["hr_10"],
                "val/mrr_10": val_ranking_metrics["mrr_10"],
                "val/precision_10": val_ranking_metrics["precision_10"],
                "val/recall_10": val_ranking_metrics["recall_10"],
                "val/map_10": val_ranking_metrics["map_10"],
                "val/ndcg_10": val_ranking_metrics["ndcg_10"],
            })

            print(f"Ep {epoch+1} (Step {global_step}): "
                f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, Total loss {total_loss:.3f}, "
                f"Val HR@10 {val_ranking_metrics['hr_10']:.3f}, Val MRR@10 {val_ranking_metrics['mrr_10']:.3f}, "
                f"Val Precision@10 {val_ranking_metrics['precision_10']:.3f}, Val Recall@10 {val_ranking_metrics['recall_10']:.3f}, "
                f"Val MAP@10 {val_ranking_metrics['map_10']:.3f}, Val NDCG@10 {val_ranking_metrics['ndcg_10']:.3f}, "
                f"LR {scheduler.get_last_lr()[0]:.2e}")

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping. Best val loss: {:.3f}".format(early_stopping.best_score))
                break

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

        # Finish the run and upload any remaining data.
        run.finish()


        #early_stopping.load_best_model(model)

        # Save the model with help from utils.py
        #save_model(model=model,
        #                target_dir=cfg_model["folder"],
        #                model_name=model_filename)