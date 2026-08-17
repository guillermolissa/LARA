import os
import logging
import argparse
import polars as pl
from sympy import fraction
from log import log
from utils import load_config


def load_parquet_file(filepath: str) -> pl.DataFrame:
    """Load a Parquet file using Polars."""
    return pl.read_parquet(filepath)

def merge_parquet_files(file_list: list[str]) -> pl.DataFrame:
    """Merge multiple Parquet files into a single Parquet file."""
    dataframes = [load_parquet_file(file) for file in file_list]
    merged_df = pl.concat(dataframes)
    return merged_df



def split_merged_train_test(df: pl.DataFrame, fraction: float = 0.2, seed: int = 42) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a Polars DataFrame into train and test sets."""

    dataset2split = add_row_id(df)  # Add row_id for splitting
    test_df = dataset2split.sample(fraction=fraction, with_replacement=False, seed=seed)
    test_ids = test_df.select("row_id").to_series().to_list()
    train_df = dataset2split.filter(~pl.col("row_id").is_in(test_ids))

    return train_df.drop("row_id"), test_df.drop("row_id")

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
def main(train_file_name: str, test_file_name: str, seed: int, fraction: float):
    """Main function to load, split and save the data."""

    
    print("Init data splitter")
    print("Loading configuration")    
    # Load config
    cfg_data = load_config("config/config.yaml")["data"]
    train_folder_path =  cfg_data["train"] # Replace with your actual file path
    train_filepath = os.path.join(train_folder_path, train_file_name)

    test_folder_path =  cfg_data["test"] # Replace with your actual file path
    test_filepath = os.path.join(test_folder_path, test_file_name)
    
    #Open a file safely, raising an error if the file does not exist.
    if not os.path.exists(train_filepath):
        raise FileNotFoundError(f"File not found: {train_filepath}")
    elif not os.path.exists(test_filepath):
        raise FileNotFoundError(f"File not found: {test_filepath}")

 
    print("Splitting data into train and test sets")
    df_merged = merge_parquet_files([train_filepath, test_filepath])
    
    df_train, df_test = split_merged_train_test(df_merged, fraction=fraction, seed=seed)

    print("Train shape:", df_train.shape)
    print("Test shape:",  df_test.shape)
    
    # save final dataframe
    print("Saving splitted files")
    df_train.write_parquet(
        os.path.join(cfg_data["train"], train_file_name.split(".")[0] + "_reordered.parquet"),
        use_pyarrow=True
    )
    
    df_test.write_parquet(
        os.path.join(cfg_data["test"], test_file_name.split(".")[0] + "_reordered.parquet"),
        use_pyarrow=True
    )
    
    print("Data splitter completed.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to load, split and save the data.')
    parser.add_argument('-t', '--train_file_name', type=str, required=True, help='Training file to split')
    parser.add_argument('-T', '--test_file_name', type=str, required=True, help='Test file to split')
    parser.add_argument('-s', '--seed', type=int, required=False, help='Seed for random number generator.', default=42)
    parser.add_argument('-p', '--fraction', type=float, required=False, help='Fraction of data to be used for testing.', default=0.2)

    args = parser.parse_args()
    
    main(
        train_file_name=args.train_file_name,
        test_file_name=args.test_file_name,
        seed=args.seed,
        fraction=args.fraction
    )
