import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Keep only accelerometer columns")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="processed_data.csv",
        help="Path to the processed CSV (default: processed_data.csv)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="processed_data_acc.csv",
        help="Path to write accelerometer‐only CSV (default: processed_data_acc.csv)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Could not find input file: {args.input_csv}")

    print(f"Loading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    # Collect all columns ending in '_acc_x', '_acc_y', or '_acc_z'
    accel_cols = [c for c in df.columns if c.endswith(("_acc_x", "_acc_y", "_acc_z"))]
    print(f"Detected accelerometer columns: {accel_cols}")

    # Also keep 'timestamp', 'sbj_id', and 'activity' if they exist
    keep = []
    for col in ['timestamp', 'sbj_id', 'activity']:
        if col in df.columns:
            keep.append(col)

    selected = keep + accel_cols
    print(f"Filtering to columns: {selected}")

    df_acc = df[selected]
    df_acc.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv} with shape {df_acc.shape}")

if __name__ == "__main__":
    main()