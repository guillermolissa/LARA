# Import libraries
import os
import polars as pl
import pyarrow as pa
import time
from utils import load_config



fields = [
    pa.field('session_id', pa.int64()),
    pa.field('item_id', pa.int64()),
    pa.field('date', pa.timestamp('ms'))
]


def main(subset="train", config_path="config/config.yaml") -> None:
    """Main function for data loading script."""
    # Load config
    cfg_data = load_config(config_path)
    
    input_data_path = cfg_data["data"]["raw"]
    output_data_path = cfg_data["data"]["intermediate"]
            
    assert subset in ["train", "test"], f"Invalid subset: {subset}. Allowed options are 'train' or 'test'."
    
    if subset=="train":
        print(f"Loading train folder")
        try:
            load_train(input_data_path, output_data_path)
        except Exception as e: 
            
            print(f"Error occurred while loading Train: {e}") 
    else:
        print(f"Loading test folder")
        
        try:
            load_test(input_data_path, output_data_path)
        except Exception as e: 
            
            print(f"Error occurred while loading Test: {e}") 
            print(e)



def load_train(input_data_path, output_data_path):
    
    train_session_path = os.path.join(input_data_path, "train_sessions.csv")
    train_purchases_path = os.path.join(input_data_path, "train_purchases.csv")
    
    
    start_time = time.time()
    df_train_sessions = pl.read_csv(train_session_path).with_columns(
        pl.col('date').str.slice(0,19).str.to_datetime(format="%Y-%m-%d %H:%M:%S").alias('date'),
    ).with_columns(pl.lit('session').alias('origin'))
    
    df_train_purchases = pl.read_csv(train_purchases_path).with_columns(
        pl.col('date').str.slice(0,19).str.to_datetime(format="%Y-%m-%d %H:%M:%S").alias('date'),
    ).with_columns(pl.lit('purchase').alias('origin'))
    
    # Combine both dataframes into one
    df_train = pl.concat([df_train_sessions, df_train_purchases])
    
    # save as parquet
    df_train.write_parquet(os.path.join(output_data_path, "train.parquet"))    
        
    save_time = time.time() - start_time
    
    print(f"Load train CSVs files and convert to parquet - time: {save_time:.2f} seconds")
    pass


def load_test(input_data_path, output_data_path):
    
    test_session_path = os.path.join(input_data_path, "test_full_sessions.csv")
    test_purchases_path = os.path.join(input_data_path, "test_full_purchases.csv")
    
    
    start_time = time.time()
    df_test_sessions = pl.read_csv(test_session_path).with_columns(
        pl.col('date').str.slice(0,19).str.to_datetime(format="%Y-%m-%d %H:%M:%S").alias('date'),
    ).with_columns(pl.lit('session').alias('origin'))
    
    df_test_purchases = pl.read_csv(test_purchases_path).with_columns(
        pl.col('date').str.slice(0,19).str.to_datetime(format="%Y-%m-%d %H:%M:%S").alias('date'),
    ).with_columns(pl.lit('purchase').alias('origin'))
    
    # Combine both dataframes into one
    df_test = pl.concat([df_test_sessions, df_test_purchases])
    
    # save as parquet
    df_test.write_parquet(os.path.join(output_data_path, "test.parquet"))    
        
    
    save_time = time.time() - start_time
    
    print(f"Load test CSVs files and convert to parquet - time: {save_time:.2f} seconds")
    pass
        

if __name__ == "__main__":
    main(subset="train", config_path="config/config.yaml")
    main(subset="test", config_path="config/config.yaml")
