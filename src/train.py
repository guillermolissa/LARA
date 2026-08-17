import os
import warnings
from pathlib import Path
import torch
import engine
from dataset import DressipiDataset
from torch.utils.data import DataLoader , random_split
from custom_collate import train_collate_fn, test_collate_fn
from functools import partial
#from data_setup import create_dataloaders
from utils import set_seed, load_config, save_model, build_model_name
import model_builder   # Adjust the import based on the actual location of GPTModel
from torchinfo import summary
from tokenizer import get_tokenizer
from tokenizers import Tokenizer
import torch.multiprocessing as mp
warnings.filterwarnings("ignore")  # To ignore user warnings

# Setup hyperparameters
#NUM_WORKERS = os.cpu_count()
#print("NUM_WORKERS: ", NUM_WORKERS)


def load_config_file(path: str):
    """Load a YAML configuration file."""
    # Load config for experiment
    cfg_model = load_config(path)["model"]
    cfg_data = load_config(path)["data"]
    cfg_hyperparam = load_config(path)["hyperparameters"]

    return cfg_model, cfg_data, cfg_hyperparam




# # Create DataLoaders with help from data_setup.py
# train_dataloader, val_dataloader, _ = create_dataloaders(
#     file_path=Path(cfg_data["feature_store_dir"], "train", cfg_data["feature_store_name"].rsplit('.', 1)[0] + ".parquet"),
#     tokenizer_path=cfg_data["tokenizer_path"],
#     seq_len=cfg_model["context_length"],
#     batch_size=cfg_hyperparam["batch_size"]
# )

def load_data(file_path: str,  tokenizer: Tokenizer, validation_ratio: float=0.1):
    

    train_ratio = 1 - validation_ratio


    print("Loading dataset...")
    # It only has the train split, so we divide it overselves
    ds = DressipiDataset(file_path=file_path, tokenizer=tokenizer) 
        
    # Keep 90% for training, 10% for validation
    train_ds_size = int(train_ratio * len(ds))
    val_ds_size = len(ds) - train_ds_size
    train_ds, val_ds = random_split(ds, [train_ds_size, val_ds_size])
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    
    return train_ds, val_ds


#train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
#val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)



if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)

    cfg_model, cfg_data, cfg_hyperparam = load_config_file("config/config.yaml")
    
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
    
    
    # load dataset
    file_path = Path(cfg_data["train"], cfg_data["train_feature_store"].rsplit('.', 1)[0] + ".parquet")
    tokenizer_path = cfg_data["tokenizer_path"]
    
    
    # Load tokenizers
    tokenizer = get_tokenizer(tokenizer_path=tokenizer_path)

    # Load dataset
    train_ds, val_ds = load_data(file_path=file_path, tokenizer=tokenizer, validation_ratio=validation_ratio)


    seq_len = cfg_model["context_length"]
    batch_size = cfg_hyperparam["batch_size"]
    cfg_model["items_size"] = tokenizer.get_vocab_size()
    cfg_model["item_meta_embedding"] = cfg_data["item_meta_embedding"]
    cfg_model["item_meta_id_map"] = cfg_data["item_meta_id_map"]


    customized_collate_fn = partial(
        train_collate_fn,
        device=device,
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
        model = model_builder.GPTAdaSoftmaxModel(cfg_model, tokenizer=tokenizer).to(device)
    else:
        print("Using Linear Softmax")
        model = model_builder.GPTModel(cfg_model, tokenizer=tokenizer).to(device)

        # Set loss and optimizer
        #loss_fn = torch.nn.CrossEntropyLoss()

    # TODO: Set optimizer from config file
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=cfg_hyperparam["learning_rate"], 
                                weight_decay=cfg_hyperparam["weight_decay"])

  
    #summary(model, input_size=input.shape)
    
    # Start training with help from engine.py
    model_filename = build_model_name(cfg_model, cfg_hyperparam)
    engine.train_model(model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    checkpoint_path=Path(cfg_model["folder"], model_filename.replace(".pth", "_checkpoint.pth")),
                    optimizer=optimizer,
                    device=device,
                    epochs=cfg_hyperparam["num_epochs"],
                    evaluation_frequency=cfg_hyperparam["evaluation_frequency"],
                    evaluation_num_batches=cfg_hyperparam["evaluation_num_batches"],
                    num_accumulation_steps=cfg_hyperparam["num_accumulation_steps"],
                    use_adaptive_softmax=cfg_hyperparam["use_adaptive_softmax"],
                    warmup_steps=cfg_hyperparam.get("warmup_steps", 0),
                    max_grad_norm=cfg_hyperparam.get("max_grad_norm", 1.0),
                    label_smoothing=cfg_hyperparam.get("label_smoothing", 0.1),
                    )

    # Save the model with help from utils.py
    save_model(model=model,
                    target_dir=cfg_model["folder"],
                    model_name=model_filename)