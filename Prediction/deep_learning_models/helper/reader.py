import os
import glob
import pandas as pd
import pyreadr
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# --- helpers ---
def read_rds_df(path: str) -> pd.DataFrame:
    res = pyreadr.read_r(path)
    if not res:
        raise ValueError(f"No objects in RDS: {path}")
    key = next(iter(res.keys()))
    df = res[key]
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"First object in {path} is {type(df)}, not DataFrame")
    return df

def read_many_rds(paths) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = read_rds_df(p)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("read_many_rds: no RDS files given")
    return pd.concat(dfs, axis=0, ignore_index=True)

def load_survival_dataset(index_event: str, outcome_event: str, data_root: str = None):
    idx_dir = os.path.join(data_root, index_event)
    print(idx_dir)
    y_dir  = os.path.join(idx_dir, outcome_event)
    print(y_dir)

    # sanity prints
    print(f"  [dbg] idx_dir exists: {os.path.isdir(idx_dir)} -> {idx_dir}")
    print(f"  [dbg] y_dir  exists: {os.path.isdir(y_dir)}  -> {y_dir}")

    # X shards (index-level)
    x_train_glob = os.path.join(idx_dir, "X_train", "*.rds")
    x_test_glob  = os.path.join(idx_dir, "X_test",  "*.rds")
    x_train_paths = sorted(glob.glob(x_train_glob))
    x_test_paths  = sorted(glob.glob(x_test_glob))

    print(f"  [dbg] glob X_train: {x_train_glob} -> {len(x_train_paths)} files")
    print(f"  [dbg] glob X_test : {x_test_glob}  -> {len(x_test_paths)} files")

    if not x_train_paths:
        raise FileNotFoundError(f"No X_train shards at {x_train_glob}")
    if not x_test_paths:
        raise FileNotFoundError(f"No X_test shards at {x_test_glob}")

    X_train = read_many_rds(x_train_paths)
    X_test  = read_many_rds(x_test_paths)

    # clean like the R code
    # drop all-NA columns
    X_train = X_train.loc[:, X_train.columns[X_train.notna().any(axis=0)]]
    X_test  = X_test.loc[:,  X_test.columns[X_test.notna().any(axis=0)]]
    # drop columns starting with "exo"
    X_train = X_train.loc[:, [c for c in X_train.columns if not c.startswith("exo")]]
    X_test  = X_test.loc[:,  [c for c in X_test.columns  if not c.startswith("exo")]]
    # keep rows with non-null id
    if "id" not in X_train.columns or "id" not in X_test.columns:
        raise KeyError("Column 'id' must exist in X dataframes after reading.")
    X_train = X_train[X_train["id"].notna()]
    X_test  = X_test[X_test["id"].notna()]

    # y files (index+outcome)
    y_train_path = os.path.join(y_dir, "y_train.rds")
    y_test_path  = os.path.join(y_dir, "y_test.rds")
    print(f"  [dbg] y_train.rds exists: {os.path.isfile(y_train_path)} -> {y_train_path}")
    print(f"  [dbg] y_test.rds  exists: {os.path.isfile(y_test_path)}  -> {y_test_path}")

    if not os.path.isfile(y_train_path):
        raise FileNotFoundError(f"Missing {y_train_path}")
    if not os.path.isfile(y_test_path):
        raise FileNotFoundError(f"Missing {y_test_path}")

    y_train = read_rds_df(y_train_path)
    y_test  = read_rds_df(y_test_path)

    # keep rows with non-null id
    if "id" not in y_train.columns or "id" not in y_test.columns:
        raise KeyError("Column 'id' must exist in y dataframes.")
    y_train = y_train[y_train["id"].notna()]
    y_test  = y_test[y_test["id"].notna()]

    # inner join by id (labels left, like the R return(inner_join(y, X, by="id")))
    train = pd.merge(y_train, X_train, on="id", how="inner")
    test  = pd.merge(y_test,  X_test,  on="id", how="inner")

    # enforce identical schemas
    shared_cols = [c for c in train.columns if c in test.columns]
    train = train[shared_cols].copy()
    test  = test[shared_cols].copy()

    return train, test