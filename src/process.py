# Import libraries
import os
import polars as pl
from utils import load_config
from argparse import Namespace
from datetime import datetime
import pickle
from polars import DataFrame



def filter_data(log_path: str, min_seq_len=5, min_items_freq=5) -> DataFrame:

    item_filter = pl.scan_parquet(log_path).select("session_id", "item_id").group_by("item_id")\
        .agg(pl.col("session_id").count().alias("count")).filter(pl.col("count") > min_items_freq).select("item_id").collect()
    
    
    query = (
        pl.scan_parquet(log_path)
        .select("session_id", "item_id", "date", "origin").with_columns(pl.col("date").dt.truncate("1mo").alias("period"))
        .filter((pl.col("date").is_between(datetime.strptime('2020-01-01', '%Y-%m-%d').date(), datetime.strptime('2021-07-01', '%Y-%m-%d').date())))

    )
    
    df_sessions_filtered = query.collect()

    # remove low frequency items
    df_session_low_freq_items_filtered = df_sessions_filtered.join(item_filter, on="item_id", how="inner")


    df_session_low_freq_items_filtered = df_session_low_freq_items_filtered.with_columns(
                                pl.col("item_id").count().over("session_id").alias("session_length")
    )

    # filter sessions by min length after remove low frequency items
    df_session_length_filtered = df_session_low_freq_items_filtered.filter(pl.col("session_length") >= min_seq_len)
    
    
    
    df_session_sorted = df_session_length_filtered.sort([ "session_id", "date"], descending=[False, False])
    
    
    
    df_sessions_position = df_session_sorted.with_columns(
                                pl.col("item_id").cum_count().over("session_id").alias("session_position")
    )


    return df_sessions_position.select("period", "session_id", "date", "session_position", "session_length", "item_id", "origin")

def collect_session_items(df: DataFrame) -> DataFrame:
    return df.sort("session_id", "date", "session_position")\
        .group_by("period", "session_id","session_length")\
        .agg(pl.col("item_id").alias("item_ids")
            ,pl.min("date").alias("session_start_date"), pl.max("date").alias("session_end_date"))\

def process_file_plane(log_path: str, min_seq_len: int, min_items_freq: int) -> DataFrame:
    data_filtered = filter_data(log_path, min_seq_len=min_seq_len, min_items_freq=min_items_freq)
    data_processed = data_filtered.select("session_id", "date", "session_position", "item_id").sort("session_id", "date", "session_position")
    return data_processed


def process_file(log_path: str, min_seq_len: int, min_items_freq: int) -> DataFrame:
    data_filtered = filter_data(log_path, min_seq_len=min_seq_len, min_items_freq=min_items_freq)
    #print(data_filtered.head())
    data_processed = collect_session_items(data_filtered)
    return data_processed



def load_dict(path):
    with open(path, 'rb') as handle:
        dict_loaded = pickle.load(handle)
    return dict_loaded
    
    

def main(subset) -> None:
    cfg_loaded = load_config(config_dir="config/config.yaml")
    cfg = Namespace(**cfg_loaded["data"])

    input_folder_path = cfg.intermediate
    output_folder_path = cfg.processed
    #features_data_path = cfg.features
     
    
    print("DataPipeline initialized")
    
    assert subset in ["train", "test"], f"Invalid subset: {subset}. Allowed options are 'train' or 'test'."
    
    
    
    
    print(f"Data transformation started. Processing {subset} data...")
    try:  
        input_file_path = os.path.join(input_folder_path,f"{subset}.parquet")
        output_file_path = os.path.join(output_folder_path,f"{subset}.parquet")
        
        
        
        print(f"Processing file {input_file_path}")
        df_processed = process_file(input_file_path, min_seq_len=cfg.min_session_length, min_items_freq=cfg.items_frequency)
        
        #df_processed = df_processed.with_columns(pl.col("items_ids").apply(len).alias("seq_len"))
        #df_filtered = df_processed.filter(pl.col("seq_len")>10) # keep only those sequence with more than 10 items

        # it casts item ids from str to int
        df_tosave = df_processed.with_columns(pl.col("item_ids").list.eval(pl.element().cast(pl.String)).alias("sequence_item_ids"))\
                                .drop(["item_ids"])

        # save feature store
        print(f"Saving processed file {output_file_path}")
        print(f"Number of sessions: {df_tosave.shape[0]}")
        df_tosave.write_parquet(
            output_file_path,
            use_pyarrow=True
        )

        # this file is used only to evaluete with alternative solutions, like top-k rank, that don't use the model
        df_plane_processed = process_file_plane(input_file_path, min_seq_len=cfg.min_session_length, min_items_freq=cfg.items_frequency)
        print(f"Saving processed file {output_file_path.replace(".parquet", ".txt")}")
        print(f"Number of sessions: {df_plane_processed.select('session_id').n_unique()}")

        df_plane_processed.write_csv(
                    output_file_path.replace(".parquet", ".txt"),
                   include_header=True, separator="\t")

        
        
        print("Data transformation completed.")
        
        
        
    except Exception as e: 
        
        print("Error occurred while processing Train") 
        print(e)
    


    
if __name__ == "__main__":
    main(subset="train")
    main(subset="test")
    