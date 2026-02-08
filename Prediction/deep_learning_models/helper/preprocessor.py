import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from mrmr import mrmr_classif
from sklearn.model_selection import train_test_split

def preprocess(
    train_df_with_labels: pd.DataFrame,
    test_df_with_labels: pd.DataFrame,
    uncensored_ratio: float = 0.055,
    min_var: float = 1e-5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses data for the benchmark.
    Now expects both train and test dataframes WITH labels (timeDiff, status).
    Returns: X_train, X_test, y_train, y_test
    """
    
    # ---- split targets / features (train) ----
    y_train = train_df_with_labels[["timeDiff", "status"]]
    train_features = train_df_with_labels.drop(columns=["timeDiff", "status"])

    # ---- split targets / features (test) ----
    y_test = test_df_with_labels[["timeDiff", "status"]]
    test_features = test_df_with_labels.drop(columns=["timeDiff", "status"])

    # ---- drop meta cols (same as before) ----
    cols_to_drop = ["id", "user", "pool", "Index Event", "Outcome Event", "type", "timestamp"]
    train_features = train_features.drop(columns=cols_to_drop, errors="ignore")
    test_features  = test_features.drop(columns=cols_to_drop,  errors="ignore")

    # ---- handle categoricals (fit on train, apply to test) ----
    categorical_cols = train_features.select_dtypes(include=["object", "category"]).columns
    # Ensure all categorical columns are treated as object for replacement
    train_features[categorical_cols] = train_features[categorical_cols].astype("object")
    test_features[categorical_cols]  = test_features[categorical_cols].astype("object")

    for col in categorical_cols:
        top_categories = train_features[col].value_counts().nlargest(10).index
        train_features[col] = train_features[col].where(train_features[col].isin(top_categories), "Other")
        # apply same capping to test
        test_features[col] = test_features[col].where(test_features[col].isin(top_categories), "Other")

    # ---- one-hot encode (train as template) ----
    train_features_encoded = pd.get_dummies(
        train_features, columns=categorical_cols, dummy_na=True, drop_first=True
    )
    test_features_encoded = pd.get_dummies(
        test_features, columns=categorical_cols, dummy_na=True, drop_first=True
    )

    # Align test to train columns (fill missing with 0)
    train_cols = train_features_encoded.columns
    test_features_aligned = test_features_encoded.reindex(columns=train_cols, fill_value=0)

    # ---- scale numeric columns (fit on train, apply to test) ----
    numerical_cols = train_features_encoded.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_encoded[numerical_cols])
    test_features_scaled  = scaler.transform(test_features_aligned[numerical_cols])

    # Rewrap as DataFrames and fill any residual NaNs with 0 (as before)
    X_train = pd.DataFrame(
        train_features_scaled, index=train_features_encoded.index, columns=numerical_cols
    ).fillna(0)
    X_test = pd.DataFrame(
        test_features_scaled, index=test_features_aligned.index, columns=numerical_cols
    ).fillna(0)

    # ---- drop zero-variance columns based on train; apply same mask to test ----
    cols_to_keep = X_train.columns[X_train.var() > min_var]
    X_train = X_train[cols_to_keep]
    X_test  = X_test[cols_to_keep]
    
    # ---- balance training data so events make up >= 10% ----
    event_mask = y_train["status"] == 1
    cens_mask  = y_train["status"] == 0

    event_count = event_mask.sum()
    cens_count  = cens_mask.sum()
    total = len(y_train)
    current_ratio = event_count / total

    target_ratio = uncensored_ratio  # desired minimum share of events
    if event_count == 0:
        raise ValueError("No events in training data. Cannot downsample censored.")

    if current_ratio < target_ratio and event_count > 0:
        # compute number of censored samples to keep
        max_cens_to_keep = int(event_count * (1 - target_ratio) / target_ratio)
        max_cens_to_keep = min(max_cens_to_keep, cens_count)
        
        keep_cens_idx = np.random.choice(
            y_train[cens_mask].index,
            size=max_cens_to_keep,
            replace=False
        )
        keep_idx = np.concatenate([y_train[event_mask].index, keep_cens_idx])

        X_train = X_train.loc[keep_idx].reset_index(drop=True)
        y_train = y_train.loc[keep_idx].reset_index(drop=True)
        
        new_ratio = y_train["status"].mean()
        print(f"[dbg] Downsampled censored: {cens_count} → {max_cens_to_keep}, "
            f"event ratio: {new_ratio:.3f}")
    else:
        print(f"[dbg] Skipped downsampling (event ratio {current_ratio:.3f} ≥ {target_ratio})")
        
    # ---- clean up survival targets ----
    SECONDS_IN_DAY = 86400
    mask_train = y_train["timeDiff"] > 0
    mask_test  = y_test["timeDiff"] > 0

    X_train = X_train.loc[mask_train].reset_index(drop=True)
    y_train = y_train.loc[mask_train].reset_index(drop=True)

    X_test  = X_test.loc[mask_test].reset_index(drop=True)
    y_test  = y_test.loc[mask_test].reset_index(drop=True)

    y_train["timeDiff"] /= SECONDS_IN_DAY
    y_test["timeDiff"]  /= SECONDS_IN_DAY
    return X_train, X_test, y_train, y_test

def prepare_sample_datasets_mrmr(index, outcome, sample=0.95, val_frac=0.2, feature_count=30, datasets=None):
    train_df, test_df = datasets[(index, outcome)]
    X_train, X_test, y_train, y_test = preprocess(train_df, test_df)

    # ---- MRMR feature selection ----
    target_for_fs = y_train["status"].astype(int).values
    selected_features = mrmr_classif(X=X_train, y=target_for_fs, K=feature_count)
    print("Selected features:", selected_features)

    # Subset both train and test
    X_train = X_train[selected_features]
    X_test  = X_test[selected_features]

    # ---- optional: downsample train ----
    train_idx = np.random.choice(len(X_train), math.floor(len(X_train) * sample), replace=False)
    X_train = X_train.iloc[train_idx].reset_index(drop=True)
    y_train = y_train.iloc[train_idx].reset_index(drop=True)

    # ---- split train into train/val (80/20 by default) ----
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=42, stratify=y_train["status"]
    )

    # ---- final safety: rebuild arrays ----
    durations_train = y_train_split["timeDiff"].values
    events_train    = y_train_split["status"].values.astype(bool)

    durations_val   = y_val_split["timeDiff"].values
    events_val      = y_val_split["status"].values.astype(bool)

    durations_test  = y_test["timeDiff"].values
    events_test     = y_test["status"].values.astype(bool)

    print("Train shape:", X_train_split.shape, y_train_split.shape)
    print("Val shape:", X_val_split.shape, y_val_split.shape)
    print("Test shape:", X_test.shape, y_test.shape)
    print("Train durations (days): min", durations_train.min(), "max", durations_train.max())
    print("Val durations (days): min", durations_val.min(), "max", durations_val.max())
    print("Test durations (days): min", durations_test.min(), "max", durations_test.max())

    return (
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        X_test, y_test,
        durations_train, events_train,
        durations_val, events_val,
        durations_test, events_test
    )