import polars as pl
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Top-K Rank Item Sequence Generator')
    parser.add_argument('-ftr', '--train_file_name', type=str, required=True, help='Input train file path.')
    parser.add_argument('-ft', '--test_file_name', type=str, required=True, help='Input test file path.')
    parser.add_argument('-fo', '--output_file_name', type=str, required=True, help='Output submission file path.')
    parser.add_argument('-l', '--limit', type=int, default=10, required=False, help='Limit for the generated sequence length.')    
    parser.add_argument('-rt', '--random_topk', type=bool, default=False, required=False, help='Use random top-k selection.')
    
    args = parser.parse_args()
    
    df_train = pl.read_parquet(args.train_file_name)
    df_test = pl.read_parquet(args.test_file_name)
    
    df_unpivoted = df_train.explode("sequence_item_ids").select(["session_id", 
                                                                  pl.col("sequence_item_ids").alias("item_id")])    
    df_top_items = df_unpivoted.group_by("item_id").agg(pl.col("session_id").len().alias("session_count")).sort("session_count", descending=True)
    
    # Rank 'Speed' within each 'Type 1' group
    df_ranked = df_top_items.with_row_index("rank", offset=1)
    
    df_pred_topk = None
    
    if args.random_topk:
        df_topk =df_ranked.filter(pl.col("rank") <= 50)
        topk_items = df_topk.select("item_id").to_series().to_list()

        topk_items_selected = []
        
        for _ in range(len(df_test)):
        
            topk_items_selected.append(random.sample(
                topk_items,
                k=args.limit))
        
        df_pred_topk = df_test.with_columns(pl.col("sequence_item_ids").list.slice(10, 5).alias("target_items"))\
                            .with_columns(pl.Series("predicted_items", topk_items_selected))\
                            .with_columns(pl.col("sequence_item_ids").list.slice(0, 10).alias("input_items"))
        
    else:
        df_topk =df_ranked.filter(pl.col("rank") <= args.limit)
        topk_items_selected = df_topk.select("item_id").to_series().to_list()
    
        df_pred_topk = df_test.with_columns(pl.col("sequence_item_ids").list.slice(10, 5).alias("target_items"))\
                            .with_columns(pl.lit(topk_items_selected).alias("predicted_items"))\
                            .with_columns(pl.col("sequence_item_ids").list.slice(0, 10).alias("input_items"))
    
    df_pred_topk =df_pred_topk.select([
        "input_items",
        "target_items",
        "predicted_items"
    ])
    
    df_pred_topk.write_parquet(args.output_file_name)