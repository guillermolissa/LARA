"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

from dataset import DressipiDataset, DressipiTestDataset
from torch.utils.data import DataLoader, random_split
from tokenizer import get_tokenizer
from typing import Dict, List, Tuple
from tokenizers import Tokenizer


def create_dataloaders(
    file_path: str, 
    tokenizer_path: str, 
    seq_len: int,
    batch_size: int, 
    train_size: float=0.9,
    num_workers: int=0
) -> Tuple[DataLoader, DataLoader, Tokenizer]:
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
    file_path: Path to dataset directory.
    tokenizer_path: tokenizer to perform on training and testing data.
    seq_len: Length of the sequence to be used for training and validation.
    batch_size: Number of samples per batch in each of the DataLoaders.
    train_size: Represent the proportion of the dataset to include in the train split. Value should be between 0.0 and 1.0
    num_workers: An integer for number of workers per DataLoader.

    Returns:
    A tuple of (train_dataloader, val_dataloader, Tokenizer).
    Where Tokenizer is a objecto to convert items into decimal.
    Example usage:
        train_dataloader, val_dataloader, tokenizer = create_dataloaders
    """
    # Load tokenizers
    tokenizer = get_tokenizer(tokenizer_path=tokenizer_path)
    # It only has the train split, so we divide it overselves
    ds = DressipiDataset(file_path=file_path, tokenizer=tokenizer, seq_len=seq_len) 
      
    # Keep 90% for training, 10% for validation
    train_ds_size = int(train_size * len(ds))
    val_ds_size = len(ds) - train_ds_size
    train_ds, val_ds = random_split(ds, [train_ds_size, val_ds_size])

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)


    return train_dataloader, val_dataloader, tokenizer
    

def create_test_dataloader(
    file_path: str, 
    tokenizer_path: str,
    seq_len: int,
    n_first: int,
    batch_size: int,
    num_workers: int=0
):
    """Creates a DataLoader for the test set.

    Args:
    file_path: Path to testing directory.
    tokenizer_path: tokenizer to perform on testing data.
    seq_len: Length of the sequence to be used for testing.
    n_first: Number of first tokens to be used as input.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.
    Returns:
    A DataLoader object for the test set.
    Example
    usage:
        test_dataloader = create_dataloader
    """
    
    # Load tokenizers
    tokenizer = get_tokenizer(tokenizer_path=tokenizer_path)
    
    
    ds = DressipiTestDataset(file_path=file_path, tokenizer=tokenizer, seq_len=seq_len, n_first=n_first) 
    dataloader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return dataloader
    