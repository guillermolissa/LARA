# %%
import polars as pl
import torch

# %%
df_train_init = pl.read_parquet("data/train/train.parquet")#.sample(10000, seed=42)
df_test_init = pl.read_parquet("data/test/test.parquet")#.sample(1000, seed=42)

# %%
df_train_init.shape, df_test_init.shape

# %%
df_train_init = df_train_init.with_columns(pl.col("session_id").cum_count().alias("SessionId").cast(pl.Int64))
df_test_init = df_test_init.with_columns(pl.col("session_id").cum_count().alias("SessionId").cast(pl.Int64))


# %%
last_session_id = df_train_init.select(pl.col("SessionId").max()).to_numpy()[0][0]

# %%
df_test_init = df_test_init.with_columns((pl.col("SessionId") + last_session_id).alias("SessionId"))

# %%
seed = 42
df_train_init_val = df_train_init.sample(fraction=0.01, with_replacement=False, seed=seed)
val_ids = df_train_init_val.select("SessionId").to_series().to_list()
df_train_init_tr = df_train_init.filter(~pl.col("SessionId").is_in(val_ids))

# %%
df_train_init_val.shape, df_train_init_tr.shape, df_train_init.shape

# %%
df_test_init.head()

# %%
df_train_init.head()

# %%
df_train_init_exploded = df_train_init_tr.explode("sequence_item_ids")
df_train_init_tr_exploded = df_train_init_tr.explode("sequence_item_ids")
df_train_init_val_exploded = df_train_init_val.explode("sequence_item_ids")

df_test_exploded = df_test_init.explode("sequence_item_ids")

# %%
df_train_init_tr_exploded.head()

# %%
df_train_init_val_exploded.head()

# %%
df_train_init_exploded = df_train_init_exploded.with_columns(pl.col("sequence_item_ids").cum_count().over("session_id").alias("Time").cast(pl.Float64))


# %%
df_train_init_tr_exploded = df_train_init_tr_exploded.with_columns(pl.col("sequence_item_ids").cum_count().over("session_id").alias("Time").cast(pl.Float64))
df_train_init_val_exploded = df_train_init_val_exploded.with_columns(pl.col("sequence_item_ids").cum_count().over("session_id").alias("Time").cast(pl.Float64))


df_test_exploded = df_test_exploded.with_columns(pl.col("sequence_item_ids").cum_count().over("session_id").alias("Time").cast(pl.Float64))

# %%
df_train_init_tr_exploded.head()

# %%
from tokenizer import get_tokenizer
tokenizer = get_tokenizer("data/tokenizer.json")

# %%
df_train_init_tr_exploded = df_train_init_tr_exploded.with_columns(pl.col("sequence_item_ids").map_elements(lambda x: tokenizer.token_to_id(x)).alias("ItemId").cast(pl.Int64))

# %%
df_train_init_val_exploded = df_train_init_val_exploded.with_columns(pl.col("sequence_item_ids").map_elements(lambda x: tokenizer.token_to_id(x)).alias("ItemId").cast(pl.Int64))

# %%
df_test_exploded = df_test_exploded.with_columns(pl.col("sequence_item_ids").map_elements(lambda x: tokenizer.token_to_id(x)).alias("ItemId").cast(pl.Int64))

# %%
df_train_tr = df_train_init_tr_exploded.select("SessionId", "ItemId", "Time")
df_train_valid = df_train_init_val_exploded.select("SessionId", "ItemId", "Time")
df_test = df_test_exploded.select("SessionId", "ItemId", "Time")

# %%
df_train_tr = df_train_tr.with_columns(pl.when(pl.col("ItemId").is_null()).then(tokenizer.token_to_id("[UNK]")).otherwise(pl.col("ItemId")).alias("ItemId")) 
df_train_valid = df_train_valid.with_columns(pl.when(pl.col("ItemId").is_null()).then(tokenizer.token_to_id("[UNK]")).otherwise(pl.col("ItemId")).alias("ItemId")) 
df_test = df_test.with_columns(pl.when(pl.col("ItemId").is_null()).then(tokenizer.token_to_id("[UNK]")).otherwise(pl.col("ItemId")).alias("ItemId")) 

# %%
df_train_tr.select(pl.col("SessionId").is_null().sum(), pl.col("ItemId").is_null().sum())

# %%
df_train_valid.select(pl.col("SessionId").is_null().sum(), pl.col("ItemId").is_null().sum())

# %%
df_test.select(pl.col("SessionId").is_null().sum(), pl.col("ItemId").is_null().sum())

# %% [markdown]
# ## Save final datasets

# %%
df_train_tr.select("SessionId", "ItemId", "Time").write_csv("data/train/dressiformer_train_tr.txt", include_header=True, separator="\t")
df_train_valid.select("SessionId", "ItemId", "Time").write_csv("data/train/dressiformer_train_valid.txt", include_header=True, separator="\t")

df_test.select("SessionId", "ItemId", "Time").write_csv("data/test/dressiformer_test.txt", include_header=True, separator="\t")

# %%



