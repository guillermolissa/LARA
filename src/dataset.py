import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
import pyarrow.parquet as pq
import numpy as np

def file_iterator(file_name, batch_size):
    parquet_file = pq.ParquetFile(file_name)
    for record_batch in parquet_file.iter_batches(columns=["sequence_item_ids", "sequence_skips"], batch_size=batch_size):
        yield (np.concatenate(record_batch["sequence_item_ids"].to_pandas().values).reshape(batch_size, 20),
            np.concatenate(record_batch["sequence_skips"].to_pandas().values).reshape(batch_size, 20))

class DressipiIterDatasetIter(IterableDataset):

    def __init__(self, filename):

        #Store the filename in object's memory
        self.filename = filename

        #And that's it, we no longer need to store the contents in the memory
    
    def __len__(self):
        # returns length of data
        parquet_file = pq.ParquetDataset(self.filename)
        nrows = sum(p.count_rows() for p in parquet_file.fragments)
        return nrows

    def __iter__(self):

        #Create an iterator
        file_itr = file_iterator(self.filename)

        return file_itr


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class DressipiDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        super().__init__()
        
        # Load the Parquet file
        self.data = pq.read_table(file_path, columns=["sequence_item_ids"])
        self.tokenizer_src = tokenizer
        
        # Assuming the DataFrame has features in columns and the last column is the label
        self.features = self.data['sequence_item_ids'].to_pandas().values
        

    def __len__(self):
        # returns length of data
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        src_items = self.features[idx]
        
        
        # Transform items into ids
        item_sequence_ids = []
        for item in src_items:
            item_id = self.tokenizer_src.token_to_id(item)
            if item_id is None: 
                item_sequence_ids.append(self.tokenizer_src.token_to_id("[UNK]"))
            else:
                item_sequence_ids.append(item_id)
                
        

        # Add <s> and </s> token
        return item_sequence_ids
        
          



    
if __name__ == '__main__':
    from pathlib import Path
    from tokenizer import get_tokenizer
    # Example usage
    #load tokenizer
    tokenizer_path =  Path('data/tokenizer.json') #Path(cfg_data["tokenizer_source"])
    tokenizer = get_tokenizer(tokenizer_path)
    dataset = DressipiDataset("data/train/train.parquet",tokenizer=tokenizer)
    print(len(dataset))
    print(dataset[3])