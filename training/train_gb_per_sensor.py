import os
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def train_single_sensor(feature_csv, model_output):
    df = pd.read_csv(feature_csv)
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.10, random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": 19,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "max_depth": -1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "n_estimators": 1000,
        "seed": 42,
        "verbose": -1
    }

    print(f"Training LightGBM for {feature_csv}...")
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True),
            lgb.log_evaluation(period=100)
        ]
    )

    val_pred_prob = booster.predict(X_val, num_iteration=booster.best_iteration)
    val_preds = np.argmax(val_pred_prob, axis=1)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    print(f"Validation macro-F1 for {feature_csv}: {val_f1:.4f}")

    with open(model_output, "wb") as f:
        pickle.dump(booster, f)
    print(f"Saved model to {model_output}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--models_dir",    required=True)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    for sensor in ["right_arm", "right_leg", "left_leg", "left_arm"]:
        feature_csv = os.path.join(args.features_dir, f"features_{sensor}.csv")
        model_output = os.path.join(args.models_dir, f"lgbm_{sensor}.pkl")
        train_single_sensor(feature_csv, model_output)
