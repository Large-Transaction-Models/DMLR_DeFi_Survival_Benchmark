# general imports
from datetime import datetime
import os
import shutil
import glob
import math
import random
import json
import argparse
from pathlib import Path
import sys
from typing import Tuple, Optional

# data science imports
import pandas as pd
import numpy as np
import torch
import torchtuples as tt
from pycox.models import CoxPH, DeepHitSingle, MTLR
from pycox.evaluation import EvalSurv
from lifelines.utils import concordance_index

# local imports
from helper.reader import load_survival_dataset
from helper.preprocessor import preprocess, prepare_sample_datasets_mrmr


# parse the command line arguments
parser = argparse.ArgumentParser(description="Example with arguments.")
parser.add_argument("--index", type=str, help="Index Event")
parser.add_argument("--trials", type=int, help="Number of trials")
parser.add_argument("--model", type=str, help="Model Type")
parser.add_argument("--seed", type=int, help="Random Seed", default=123)
parser.add_argument("--log", type=bool, help="Enable logging", default=True)

args = parser.parse_args()

# Convert args to normal Python variables
index_arg = args.index.title() if args.index else None
trials = args.trials
model_type = args.model.lower() if args.model else None
seed = args.seed
log_mode = args.log
    
if not (index_arg and trials and model_type):
    print("You have to choose an index with --index in title case")
    print("You have to choose the number of trials with --trials")
    print("You have to choose a model type with --model")
    exit()
    
log_file = open(f"{model_type}_{index_arg.lower()}.log", "a")
sys.stdout = log_file
sys.stderr = log_file
print()
print()
print()
print()
print()
print("Starting the script...")
print("Timestamp:", datetime.now())
print("Chosen Index:", index_arg)
print("Chosen Trials:", trials)
print("Chosen Model Type:", model_type)
print("Chosen Seed:", seed)


# --- CONFIG ---
DATA_ROOT = "/data/IDEA_DeFi_Research/Data/Survival_Data"  # <-- your actual root
INDEX_EVENTS  = [index_arg]
OUTCOME_EVENTS = ["Borrow", "Repay", "Deposit","Withdraw","Account Liquidated"]
#OUTCOME_EVENTS = ["Withdraw","Account Liquidated"]

# ---- iterate pairs (like your R calls) ----
event_pairs = [(i, o) for i in INDEX_EVENTS for o in OUTCOME_EVENTS if i != o]

datasets = {}
for index_event, outcome_event in event_pairs:
    print("\n" + "="*60)
    print(f"Loading: {index_event} -> {outcome_event}")
    print("="*60)
    try:
        train_df, test_df = load_survival_dataset(index_event, outcome_event, data_root=DATA_ROOT)
        print(f"  train: {train_df.shape} | test: {test_df.shape}")
        datasets[(index_event, outcome_event)] = (train_df, test_df)
        # break  # uncomment if you only want the first successful pair
    except Exception as e:
        print(f"  Skipping: {e}")
        continue

# check if GPU is available
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# defining the hyperparameter search space
search_space = {
    "num_layers": [2, 3, 4, 5,6,7,8,9,10,12,16,18],
    "hidden_sizes": [16, 32, 64, 128, 256,512],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "batch_size": [256, 512, 1024, 2048, 4096],
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-4],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "batch_norm": [True],
    "optimizer": ["Adam", "AdamW", "RMSprop"],  # sampled as string
}


def sample_config():
    num_layers = random.choice(search_space["num_layers"])
    hidden_sizes = ([random.choice(search_space["hidden_sizes"]) for _ in range(num_layers)])
    
    config = {
        "hidden_sizes": hidden_sizes,
        "dropout": random.choice(search_space["dropout"]),
        "batch_size": random.choice(search_space["batch_size"]),
        "weight_decay": random.choice(search_space["weight_decay"]),
        "batch_norm": random.choice(search_space["batch_norm"]),
        "optimizer": random.choice(search_space["optimizer"]),  # e.g., Adam, AdamW, RMSprop
        "num_layers": num_layers,
    }
    
    # if layers are small, you can afford a larger learning rate
    if num_layers <= 6:
        config["learning_rate"] = random.choice([1e-3, 5e-4, 1e-4])
        config["epochs"] = 5  # fewer epochs for shallower nets
    else:
        config["learning_rate"] = random.choice([5e-4, 1e-4, 5e-5, 1e-5])
        config["epochs"] = 10  # more epochs for deeper nets and slower learning rates
        
    # choose activation based on depth
    if num_layers <= 3:
        allowed_acts = [torch.nn.ReLU, torch.nn.Tanh, torch.nn.LeakyReLU]
    elif num_layers <= 8:
        allowed_acts = [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.SiLU, torch.nn.GELU]
    else:
        # Very deep nets → avoid saturating activations (Tanh, ELU) due to vanishing gradients
        allowed_acts = [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.SiLU, torch.nn.GELU]

    config["activation"] = random.choice(allowed_acts)

    return config

def decide_clip_grad(config):
    # decide whether to use gradient clipping
    use_clip = False
    clip_value = None
    
    num_layers = config["num_layers"]+2
    activation = config["activation"]
    batch_norm = config["batch_norm"]
    learning_rate = config["learning_rate"]
    optimizer = config["optimizer"]
    batch_size = config["batch_size"]

    # ---- 1. Depth-based rule ----
    if num_layers >= 8:
        use_clip = True
        clip_value = 1.0  # deeper nets → more likely to explode

    # ---- 2. Activation-based rule ----
    if activation in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU]:
        # these are unbounded → higher risk
        use_clip = True
        clip_value = clip_value or 1.0  # ensure it's at least 1.0

    if activation in [torch.nn.SELU, torch.nn.Tanh]:
        # tend to shrink gradients, so less need for clipping
        if num_layers < 10:
            use_clip = False

    # ---- 3. BatchNorm rule ----
    if batch_norm:
        # BatchNorm stabilizes activations → less need to clip unless network is very deep
        if num_layers < 12:
            use_clip = False

    # ---- 4. Learning rate rule ----
    if learning_rate >= 5e-4:
        # aggressive learning rate → higher chance of exploding gradients
        use_clip = True
        clip_value = clip_value or 1.0

    # ---- 5. Optimizer rule ----
    if optimizer == "RMSprop":
        # RMSprop can overreact to spikes; clipping helps
        use_clip = True
        clip_value = clip_value or 1.0

    # ---- 6. Batch size rule ----
    if batch_size < 512:
        # smaller batch = noisier gradients
        use_clip = True
        clip_value = clip_value or 0.5
        
    return use_clip, clip_value


def train_and_eval(config, X_train_split, X_val_split, X_test, durations_train, durations_val, durations_test, events_train, events_val, events_test):
    in_features = X_train_split.shape[1]
    net = tt.practical.MLPVanilla(
        in_features, 
        config["hidden_sizes"], 
        out_features=1,
        batch_norm=config["batch_norm"],
        dropout=config["dropout"],
        activation=config["activation"]
    ).to("cuda")

    if config["optimizer"] == "Adam":
        model = CoxPH(net, tt.optim.Adam)
    elif config["optimizer"] == "AdamW":
        model = CoxPH(net, torch.optim.AdamW)
    elif config["optimizer"] == "RMSprop":
        model = CoxPH(net, torch.optim.RMSprop)

    # Apply LR and weight decay if supported
    if hasattr(model.optimizer, "set_lr"):
        model.optimizer.set_lr(config["learning_rate"])
    if hasattr(model.optimizer, "set_weight_decay"):
        model.optimizer.set_weight_decay(config["weight_decay"])
        
    #CHANGED THIS LINE
    if hasattr(model.optimizer, "clip_grad_norm"):
        model.optimizer.clip_grad_norm = 1.0  # prevents gradient explosion

    log = model.fit(
        X_train_split.values.astype("float32"),
        (durations_train, events_train),
        batch_size=config["batch_size"],
        epochs=config.get("epochs", 5),
        val_data=(X_val_split.values.astype("float32"), (durations_val, events_val)),
        verbose=False,
        num_workers=16,
    )

    # Risk scores & c-index
    try:
        risk_scores = -model.predict(X_test.values.astype("float32"))
        risk_scores = risk_scores.flatten()  # ensure 1D array
        c_index = concordance_index(durations_test, risk_scores, events_test)

    except Exception as e:
        import traceback
        print("\n[ERROR] Failed to compute c-index.")
        print("Exception:", type(e).__name__, "-", e)
        traceback.print_exc()

        # Optional: Save failed batch to inspect later
        debug_info = {
            "risk_scores_nan": np.isnan(risk_scores).any() if "risk_scores" in locals() else None,
            "risk_scores_inf": np.isinf(risk_scores).any() if "risk_scores" in locals() else None,
            "durations_nan": np.isnan(durations_test).any(),
            "events_nan": np.isnan(events_test).any(),
            
        }
        print("Debug info:", debug_info)

        # Default to NaN score so loop continues
        c_index = -1

    return c_index, config

def train_and_eval_deephit(config, X_train_split, X_val_split, X_test, durations_train, durations_val, durations_test, events_train, events_val, events_test, num_durations=100):
    # 1) Discretize time with N bins (automatic, sorted)
    labtrans = DeepHitSingle.label_transform(num_durations)
    y_train_dh = labtrans.fit_transform(durations_train, events_train)
    y_val_dh   = labtrans.transform(durations_val,  events_val)
    duration_index = labtrans.cuts

    # 2) Net
    in_features = X_train_split.shape[1]
    net = tt.practical.MLPVanilla(
        in_features,
        config["hidden_sizes"],
        out_features=labtrans.out_features,
        batch_norm=config["batch_norm"],
        dropout=config["dropout"],
        activation=config["activation"],
    ).to("cuda")

    # 3) Model
    if config["optimizer"] == "Adam":
        model = DeepHitSingle(net, tt.optim.Adam, duration_index=duration_index)
    elif config["optimizer"] == "AdamW":
        model = DeepHitSingle(net, torch.optim.AdamW, duration_index=duration_index)
    else:
        model = DeepHitSingle(net, torch.optim.RMSprop, duration_index=duration_index)

    if hasattr(model.optimizer, "set_lr"):
        model.optimizer.set_lr(config["learning_rate"])
    if hasattr(model.optimizer, "set_weight_decay"):
        model.optimizer.set_weight_decay(config["weight_decay"])
        
    use_clip, clip_value = decide_clip_grad(config)
    if use_clip:
        if hasattr(model.optimizer, "clip_grad_norm"):
            model.optimizer.clip_grad_norm = clip_value
    config["clip_grad"] = use_clip
    config["clip_value"] = clip_value

    # 4) Fit
    log = model.fit(
        X_train_split.values.astype("float32"),
        y_train_dh,
        batch_size=config["batch_size"],
        epochs=config.get("epochs", 5),
        val_data=(X_val_split.values.astype("float32"), y_val_dh),
        verbose=False,
        num_workers=16,
    )
    print("Finish fitting DeepHitSingle Model")

    # 5) Predict + Antolini
    surv = model.predict_surv_df(X_test.values.astype("float32"))
    print("Finished predicting survival function")
    t_idx = surv.index.values
    durations_test_clipped = np.clip(durations_test, t_idx.min(), t_idx.max())
    ev = EvalSurv(surv, durations_test_clipped, events_test, censor_surv="km")
    c_index = ev.concordance_td("antolini")
    return c_index, config

log_file = Path(f"results/{model_type}_{index_arg.lower()}.json")

# Load existing results if file exists
if log_file.exists():
    with open(log_file, "r") as f:
        results = json.load(f)
else:
    results = {
        "model": f"{model_type}",
        "package": "pycox",
        "index": f"{index_arg.lower()}",
        "trials": {
            "borrow": [],
            "repay": [],
            "account liquidated": [],
            "deposit": [],
            "withdraw": [],
        },
    }
    # ensure directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # initialize empty file with json
    with open(log_file, "w") as f:
        json.dump(results, f, indent=2)
    
best_score = -1
best_config = None
random.seed(seed)

for index in INDEX_EVENTS:
    for outcome in OUTCOME_EVENTS:
        print()
        print("Starting Outcome:",outcome)
        print()
        if index == outcome:
            print(f"Skipping {index} -> {outcome} (same event)")
            continue
        
        X_train_split, y_train_split, X_val_split, y_val_split, X_test,y_test,durations_train, events_train,durations_val, events_val,durations_test, events_test = prepare_sample_datasets_mrmr(index,outcome, datasets=datasets)
        
        #print("Feature magnitude stats:", np.abs(X_train_split).max().max())
        for i in range(trials):  
            # number of random trials
            feature_count = random.choice(range(10,116,5))
            num_durations = random.choice([50, 100, 150,200,250])
            cfg = sample_config()
            
            # fetch the splits with the chosen number of features
            X_train_split, y_train_split, X_val_split, y_val_split, X_test,y_test,durations_train, events_train,durations_val, events_val,durations_test, events_test = prepare_sample_datasets_mrmr(index,outcome, datasets=datasets, feature_count=feature_count)

            print(f"\n>>> Trial {i+1}: {cfg}")

            score, cfg = train_and_eval_deephit(cfg,
                                           X_train_split, X_val_split, X_test,
                                           durations_train, durations_val, durations_test,
                                           events_train, events_val, events_test, num_durations=num_durations)
            print(f"C-index = {score:.4f}")
        
            # Record this trial safely
            trial_result = {
                "timestamp": datetime.now().isoformat(),
                "model_arch": "MLPVanilla",
                "config": {
                    **cfg,
                    "activation": cfg["activation"].__name__,  # save as string
                },
                "c_index": float(score),
                "feature_selection": "MRMR",
                "feature_count": int(X_train_split.shape[1]),
                "seed": seed,
                "clip_grad": cfg.get("clip_grad", False),
                "clip_value": cfg.get("clip_value", -1),
                "durations": num_durations,
            }
        
            results["trials"][outcome.lower()].append(trial_result)
            
        
            # Save immediately (so we don’t lose progress if interrupted)
            with open(log_file, "w") as f:
                json.dump(results, f, indent=2)
        
            # Track best score
            if score > best_score:
                best_score = score
                best_config = cfg
        print("Index:", index)
        print("Outcome:", outcome)
        print("\nBest config:", best_config)
        print("Best C-index:", best_score)
        best_score = -1
