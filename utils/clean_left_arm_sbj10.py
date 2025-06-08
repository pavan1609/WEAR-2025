import os
import numpy as np

def clean_one_subject(sensor_dir, subject_prefix):
    """
    Loads `<sensor>/<prefix>_X.npy` and `<sensor>/<prefix>_y.npy`,
    removes any window rows where X has NaN or Inf, and
    overwrites the files with the cleaned arrays.
    """
    x_path = os.path.join(sensor_dir, f"{subject_prefix}_X.npy")
    y_path = os.path.join(sensor_dir, f"{subject_prefix}_y.npy")

    print(f"Loading {x_path} and {y_path}...")
    X = np.load(x_path)  # shape (N, 50, 3)
    y = np.load(y_path)  # shape (N,)

    # Find windows that are finite and have nonzero variance
    mask_finite = np.all(np.isfinite(X), axis=(1, 2))
    var_per_window = np.var(X, axis=(1, 2))
    mask_var = var_per_window > 0
    good_mask = mask_finite & mask_var

    num_bad = np.sum(~good_mask)
    num_total = len(y)
    print(f"  → {num_bad}/{num_total} windows are bad (NaN/Inf or zero‐variance).")

    if num_bad == 0:
        print("  No cleanup needed; exiting.")
        return

    X_clean = X[good_mask]
    y_clean = y[good_mask]

    # Overwrite the files
    np.save(x_path, X_clean)
    np.save(y_path, y_clean)
    print(f"  Overwrote {x_path} and {y_path} with cleaned arrays.")
    print(f"  New shapes: X={X_clean.shape}, y={y_clean.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Clean left_arm_sbj_10 segments by dropping NaN/Inf windows"
    )
    parser.add_argument(
        "--sensor_dir",
        required=True,
        help="Path to raw_segments_by_sensor/left_arm"
    )
    parser.add_argument(
        "--subject_prefix",
        default="left_arm_sbj_10",
        help="Filename prefix for the subject to clean (default: left_arm_sbj_10)"
    )
    args = parser.parse_args()

    clean_one_subject(args.sensor_dir, args.subject_prefix)