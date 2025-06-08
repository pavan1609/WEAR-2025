import os
import numpy as np
import pandas as pd

def extract_features_for_sensor(segments_dir, feature_csv):
    all_feats, all_labels = [], []

    for fn in sorted(os.listdir(segments_dir)):
        if not fn.endswith("_X.npy"):
            continue
        prefix = fn[:-6]  # e.g. "right_arm_sbj_0"
        X = np.load(os.path.join(segments_dir, fn))
        y = np.load(os.path.join(segments_dir, prefix + "_y.npy"))

        # Basic features (mean/std/min/max per axis)
        mean = X.mean(axis=1)
        std = X.std(axis=1)
        mn = X.min(axis=1)
        mx = X.max(axis=1)

        # Optional magnitude (highly recommended)
        mag = np.linalg.norm(X, axis=2)
        mag_mean = mag.mean(axis=1).reshape(-1,1)
        mag_std = mag.std(axis=1).reshape(-1,1)
        mag_min = mag.min(axis=1).reshape(-1,1)
        mag_max = mag.max(axis=1).reshape(-1,1)

        feats = np.concatenate([mean, std, mn, mx, mag_mean, mag_std, mag_min, mag_max], axis=1)
        all_feats.append(feats)
        all_labels.append(y.reshape(-1, 1))

    X_all = np.concatenate(all_feats, axis=0)
    y_all = np.concatenate(all_labels, axis=0).ravel()

    # Column names
    col_names = []
    for stat in ["mean", "std", "min", "max"]:
        for c in range(3):
            col_names.append(f"{stat}_ch{c}")
    col_names += ["mag_mean", "mag_std", "mag_min", "mag_max"]

    df = pd.DataFrame(X_all, columns=col_names)
    df["label"] = y_all
    df.to_csv(feature_csv, index=False)
    print(f"Wrote {feature_csv} with shape {df.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    extract_features_for_sensor(args.segments_dir, args.output_csv)