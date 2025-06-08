#!/usr/bin/env python3
"""
Expected directory structure:
  raw_segments_by_sensor/
    right_arm/
      right_arm_sbj_*_X.npy
      right_arm_sbj_*_y.npy
    right_leg/, left_leg/, left_arm/

Usage:
  python training/train_gb_per_sensor.py \
    --segments_dir raw_segments_by_sensor \
    --output_dir models_by_sensor \
    --num_leaves 64 \
    --max_depth 7 \
    --learning_rate 0.05 \
    --n_estimators 500 \
    --subsample 0.8 \
    --colsample_bytree 0.8 \
    --min_child_samples 20 \
    --reg_alpha 0.1 \
    --reg_lambda 0.1 \
    --early_stopping_rounds 10
"""
import os
import argparse
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# List of sensors to train
SENSORS = ['right_arm', 'right_leg', 'left_leg', 'left_arm']


def load_sensor_segments(segments_dir, sensor):
    sensor_path = os.path.join(segments_dir, sensor)
    if not os.path.isdir(sensor_path):
        raise FileNotFoundError(f"Missing sensor directory: {sensor_path}")
    X_list, y_list = [], []
    for fname in sorted(os.listdir(sensor_path)):
        path = os.path.join(sensor_path, fname)
        if fname.endswith('_X.npy'):
            X_list.append(np.load(path))
        elif fname.endswith('_y.npy'):
            y_list.append(np.load(path))
    if not X_list or not y_list:
        raise FileNotFoundError(f"No segment files for sensor '{sensor}' in {sensor_path}")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(int)
    return X, y


def train_sensor(sensor, X, y, params, early_stopping_rounds, random_state):
    # Flatten time windows to flat features
    N, T, C = X.shape
    X_flat = X.reshape(N, T * C)
    # Split train/validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_flat, y, test_size=0.2, stratify=y, random_state=random_state
    )
    # Initialize classifier
    model = lgb.LGBMClassifier(
        objective='multiclass',
        random_state=random_state,
        n_jobs=-1,
        **params
    )
    # Train with early stopping via callbacks
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(period=0)
        ]
    )
    # Predict and score
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average='macro')
    print(f"{sensor} validation macro-F1: {f1:.4f}")
    return model


def main(segments_dir, output_dir, random_state, early_stopping_rounds, **params):
    os.makedirs(output_dir, exist_ok=True)
    for sensor in SENSORS:
        print(f"\n=== Training {sensor} ===")
        X, y = load_sensor_segments(segments_dir, sensor)
        print(f"Loaded {X.shape[0]} windows (shape {X.shape[1:]}) for {sensor}")
        model = train_sensor(sensor, X, y, params, early_stopping_rounds, random_state)
        out_path = os.path.join(output_dir, f'lgbm_{sensor}.pkl')
        joblib.dump(model, out_path)
        print(f"Saved model to {out_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Train per-sensor LightGBM with early stopping via callbacks")
    p.add_argument('--segments_dir', required=True, help='raw_segments_by_sensor directory')
    p.add_argument('--output_dir', required=True, help='Directory to save models')
    p.add_argument('--random_state', type=int, default=42)
    # Hyperparameters
    p.add_argument('--num_leaves', type=int, default=31)
    p.add_argument('--max_depth', type=int, default=-1)
    p.add_argument('--learning_rate', type=float, default=0.1)
    p.add_argument('--n_estimators', type=int, default=100)
    p.add_argument('--subsample', type=float, default=1.0)
    p.add_argument('--colsample_bytree', type=float, default=1.0)
    p.add_argument('--min_child_samples', type=int, default=20)
    p.add_argument('--reg_alpha', type=float, default=0.0)
    p.add_argument('--reg_lambda', type=float, default=0.0)
    p.add_argument('--early_stopping_rounds', type=int, default=10)
    args = p.parse_args()
    params = {
        'num_leaves': args.num_leaves,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'min_child_samples': args.min_child_samples,
        'reg_alpha': args.reg_alpha,
        'reg_lambda': args.reg_lambda
    }
    main(
        segments_dir=args.segments_dir,
        output_dir=args.output_dir,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        **params
    )
