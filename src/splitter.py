import os
import logging
import argparse
import polars as pl
from log import log
from utils import load_config


def load_parquet_file(filepath: str) -> pl.DataFrame:
    """Load a Parquet file using Polars."""
    return pl.read_parquet(filepath)

def split_train_test(df: pl.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a Polars DataFrame into train and test sets without using pandas."""
    test_df = df.sample(fraction=test_size, with_replacement=False, seed=seed)
    test_ids = test_df.select("row_id").to_series().to_list()
    train_df = df.filter(~pl.col("row_id").is_in(test_ids))
    return train_df, test_df

def split_data_on_date(df: pl.DataFrame, date_col: str, split_date: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a Polars DataFrame into train and test sets based on a date column."""
    train_df = df.filter(pl.col(date_col) < split_date)
    test_df = df.filter(pl.col(date_col) >= split_date)
    return train_df, test_df

def add_row_id(df: pl.DataFrame) -> pl.DataFrame:
    """Add a unique row ID column to help with splitting."""
    return df.with_row_index(name="row_id")

def drop_row_id(df: pl.DataFrame) -> pl.DataFrame:
    """Remove  row ID column used to split."""
    return df.drop("row_id")


@log
def main(file_name: str, seed: int, fraction: float):
    """Main function to load, split and save the data."""

    
    print("Init data splitter")
    print("Loading configuration")    
    # Load config
    cfg_data = load_config("config/config.yaml")["data"]
    folder_path =  cfg_data["sample"] # Replace with your actual file path
    filepath = os.path.join(folder_path, file_name)
    
    #Open a file safely, raising an error if the file does not exist.
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    
    print(f"Loading file: {filepath}")
    # Load the Parquet file
    df = load_parquet_file(filepath)
    df = add_row_id(df)
    
    print("Splitting data into train and test sets")
    df_train, df_test = split_train_test(df, test_size=fraction, seed=seed)
    
    # Drop the row ID column from train and test sets
    df_train = drop_row_id(df_train)
    df_test = drop_row_id(df_test)

    print("Train shape:", df_train.shape)
    print("Test shape:",  df_test.shape)
    
    # save final dataframe
    print("Saving splitted files")
    df_train.write_parquet(
        os.path.join(cfg_data["train"], file_name),
        use_pyarrow=True
    )
    
    df_test.write_parquet(
        os.path.join(cfg_data["test"], file_name),
        use_pyarrow=True
    )
    
    print("Data splitter completed.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to load, split and save the data.')
    parser.add_argument('-f', '--file_name', type=str, required=True, help='File to split into train and test')
    parser.add_argument('-s', '--seed', type=int, required=False, help='Seed for random number generator.', default=42)
    parser.add_argument('-p', '--fraction', type=float, required=False, help='Fraction of data to be used for testing.', default=0.2)

    args = parser.parse_args()
    
    main(
        file_name=args.file_name,
        seed=args.seed,
        fraction=args.fraction
    )
