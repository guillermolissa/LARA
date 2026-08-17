import warnings
import json
import os
from pathlib import Path
import pyarrow.parquet as pq
from utils import load_config
import polars as pl
from collections import Counter
# Huggingface datasets and tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from log import log, get_logger


def get_num_row(dir_path):
    # Iterate over Parquet files in the directory
    file_list = os.listdir(dir_path)
    file_list = [f for f in file_list if f.endswith(".parquet") ]
    file_list.sort()
    
    num_rows = 0
    
    for file in file_list:
        
        file_path = os.path.join(dir_path, file)
    
        # Open the Parquet file
        parquet_file = pq.ParquetFile(file_path)
        
        num_rows += parquet_file.metadata.num_rows
        
        
    
    return num_rows


def parquet_text_iterator(dir_path, column_name, batch_size=100000):
    
    # Iterate over Parquet files in the directory
    file_list = os.listdir(dir_path)
    file_list = [f for f in file_list if f.endswith(".parquet") ]
    file_list.sort()
    
    for file in file_list:
        
        file_path = os.path.join(dir_path, file)
    
        # Open the Parquet file
        table = pq.ParquetFile(file_path)
        
        # Iterate through batches in the file
        for batch in table.iter_batches(batch_size=batch_size,columns=[column_name]):
            # Extract the text column
            for row in batch.column(column_name):
                yield row.as_py()


def build_tokenizer(tokenizer_file, dir_path):
    tokenizer_path = Path(tokenizer_file)

    if not Path.exists(tokenizer_path):
        # Count item frequencies so IDs are assigned in descending frequency order.
        # AdaptiveLogSoftmaxWithLoss requires low IDs = frequent tokens.
        item_counts = Counter()
        for item in parquet_text_iterator(dir_path, "item_id_clean"):
            item_counts[item] += 1

        # Filter by min_frequency=2, then sort descending by count.
        special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
        vocab = {token: idx for idx, token in enumerate(special_tokens)}
        for item, count in item_counts.most_common():
            if count >= 2 and item not in vocab:
                vocab[item] = len(vocab)

        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.save(str(tokenizer_path))
    else:
        warnings.warn(f"Tokenizer already exists. Loading {tokenizer_file} build.")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_tokenizer(tokenizer_path) -> Tokenizer:
    tokenizer_path = Path(tokenizer_path)
    
    if Path.exists(tokenizer_path): 
        # load tokenizer if it already exists 
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer
    
    else: 
        raise FileNotFoundError("Tokenizer doesn't exists, please build it or make sure path is right.")

def tokenizer_fixer(tokenizer_path):
    with open(tokenizer_path, 'r') as file:
        data = json.load(file)
    data_str =json.dumps(data)
    data_str = data_str.replace("\\n", '')
    with open(tokenizer_path, 'w') as file:
        file.write(data_str)
    

@log
def create_tokenizer_from_data():
    # `config` contains all your script parameters in a dictionary
    cfg_loaded = load_config(config_dir="config/config.yaml")
    cfg = cfg_loaded['data']

    print("Building custom tokenizer (frequency-sorted for Adaptive Softmax compatibility)...")
    df = pl.read_parquet(cfg["tokenizer_build_path"])
    session_items = df.select("sequence_item_ids").to_series().to_list()

    # Count item frequencies so that IDs are assigned in descending frequency order.
    # AdaptiveLogSoftmaxWithLoss partitions the vocabulary by ID ranges and assumes
    # low IDs = frequent tokens. Random or alphabetical ordering makes cutoffs meaningless.
    
    item_counts = Counter()
    for sublist in session_items:
        item_counts.update(sublist)

    # Special tokens occupy the first IDs; items follow in descending frequency order.
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for item, _ in item_counts.most_common():
        if item not in vocab:
            vocab[item] = len(vocab)

    print(f"Number of custom tokens: {len(vocab)}")

    # Build tokenizer directly from the explicit vocab dict — no trainer needed.
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))

    # Save the tokenizer
    tokenizer.save(cfg["tokenizer_path"])

    # Fix tokenizer file
    tokenizer_fixer(cfg["tokenizer_path"])

    print(f"Custom tokenizer saved to {cfg['tokenizer_path']}")
    
    
    
if __name__ == "__main__":
    create_tokenizer_from_data()