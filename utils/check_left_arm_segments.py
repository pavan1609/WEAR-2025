import os
import numpy as np

def check_sensor_folder(sensor_dir):
    """
    Iterate through every *_X.npy in sensor_dir and flag any NaN/Inf in the data.
    """
    bad_files = []
    for fn in sorted(os.listdir(sensor_dir)):
        if not fn.endswith("_X.npy"):
            continue
        path = os.path.join(sensor_dir, fn)
        arr = np.load(path)  # shape: (num_windows, 50, 3)
        if np.isnan(arr).any() or np.isinf(arr).any():
            bad_files.append(fn)
    if bad_files:
        print(f"Found {len(bad_files)} files with NaN/Inf in {sensor_dir}:")
        for b in bad_files[:10]:
            print("  ", b)
    else:
        print(f"No NaN/Inf found in any X.npy under {sensor_dir}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Check left_arm segment files for NaNs or Infs"
    )
    parser.add_argument(
        "--sensor_dir",
        required=True,
        help="Path to raw_segments_by_sensor/left_arm"
    )
    args = parser.parse_args()
    check_sensor_folder(args.sensor_dir)
