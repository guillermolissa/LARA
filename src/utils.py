"""
Contains various utility functions for PyTorch model training, load config and saving.
"""
import os
from pathlib import Path
import yaml
import random
import numpy as np
import torch


def check_folder(path, point_allowed_path=False):
    split_folder = os.path.split(path)
    if not point_allowed_path:
        if '.' in split_folder[1]:
            # path is a file
            path = split_folder[0]
    if not os.path.exists(path):
        print(f'{path} folder created')
        os.makedirs(path, exist_ok=True)



def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)


def load_model(model: torch.nn.Module,
                target_dir: str,
                model_name: str,
                device: str = "cpu",
                weights_only: bool = False):
    """Loads a PyTorch model from a target directory.

        Args:
            model: A target PyTorch model to load.
            target_dir: A directory for loading the model from.
            model_name: A filename for the saved model. Should include
            either ".pth" or ".pt" as the file extension.
            device: The device to load the model to. Default is "cpu".
            weights_only: If True, only the model weights are loaded. Default is False.
    """
    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_load_path = Path(target_dir) / model_name

    # Load the model state_dict()
    print(f"[INFO] Loading model from: {model_load_path}")

    model.load_state_dict(torch.load(model_load_path, weights_only=weights_only, map_location=torch.device(device)), strict=False)
    
    
def load_config(config_dir: str):
    """Loads a yml config file from a target directory and return a dict config.

    Args:
    config_dir: A directory for loading the config file from.

    Example usage:
    load_config(config_dir="config/model.yml")
    """
    # Load config file
    config_path = Path(config_dir)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Prepend base_path to all path-like string values under `data` and `model.folder`
    base = Path(config.get("base_path", "."))
    if base != Path("."):
        for key, val in config.get("data", {}).items():
            if isinstance(val, str) and "/" in val:
                config["data"][key] = str(base / val)
        if "folder" in config.get("model", {}):
            config["model"]["folder"] = str(base / config["model"]["folder"])

    return config


def build_model_name(cfg_model: dict, cfg_hyperparam: dict) -> str:
    """Builds a descriptive model filename encoding key hyperparameters.

    Format: {model_type}_emb{emb_dim}_h{n_heads}_l{n_layers}_ctx{ctx}_dr{drop}_lr{lr}_bs{bs}_ep{ep}_{loss}.pth

    Args:
        cfg_model: The 'model' section of the config.
        cfg_hyperparam: The 'hyperparameters' section of the config.

    Returns:
        A filename string ending in '.pth'.
    """
    lr = cfg_hyperparam["learning_rate"]
    lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    loss_tag = "adas" if cfg_hyperparam.get("use_adaptive_softmax", False) else "ce"
    meta_tag = "_meta" if cfg_model.get("use_meta_embeddings", False) else ""
    return (
        f"{cfg_model['name']}"
        f"_emb{cfg_model['emb_dim']}"
        f"_h{cfg_model['n_heads']}"
        f"_l{cfg_model['n_layers']}"
        f"_ctx{cfg_model['context_length']}"
        f"_dr{cfg_model['drop_rate']}"
        f"_lr{lr_str}"
        f"_bs{cfg_hyperparam['batch_size']}"
        f"_ep{cfg_hyperparam['num_epochs']}"
        f"_{loss_tag}"
        f"{meta_tag}"
        f".pth"
    )


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)