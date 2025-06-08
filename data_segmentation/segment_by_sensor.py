import os
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Define the 4 sensors and their 3 accelerometer columns each
SENSORS = {
    'right_arm': ['right_arm_acc_x', 'right_arm_acc_y', 'right_arm_acc_z'],
    'right_leg': ['right_leg_acc_x', 'right_leg_acc_y', 'right_leg_acc_z'],
    'left_leg':  ['left_leg_acc_x', 'left_leg_acc_y', 'left_leg_acc_z'],
    'left_arm':  ['left_arm_acc_x', 'left_arm_acc_y', 'left_arm_acc_z']
}

def sliding_window_segment_sensor(df, sensor_cols, window_size=50, stride=25, label_col='activity'):
    """
    Segments a DataFrame for a single sensor (3‐axis).
    Returns:
      X: np.ndarray of shape (num_windows, window_size, 3)
      y: np.ndarray of shape (num_windows,)
    """
    data = df[sensor_cols].values      # shape (N, 3)
    labels = df[label_col].fillna(0).astype(int).values

    X, y = [], []
    N = len(data)
    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        window = data[start:end]            # (50, 3)
        window_labels = labels[start:end]   # (50,)
        counts = pd.Series(window_labels).value_counts(dropna=True)
        numeric_label = int(counts.idxmax()) if not counts.empty else 0
        X.append(window)
        y.append(numeric_label)

    return np.array(X), np.array(y)

def save_sensor_segments(X, y, output_dir, sensor, prefix):
    """
    Saves X and y for one subject & one sensor:
    {sensor}_{subject}_X.npy and {sensor}_{subject}_y.npy
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{sensor}_{prefix}_X.npy"), X)
    np.save(os.path.join(output_dir, f"{sensor}_{prefix}_y.npy"), y)
    logger.info(f"Saved {sensor} segments: {sensor}_{prefix}_X.npy, {sensor}_{prefix}_y.npy")

def segment_all_subjects_by_sensor(data_dir, window_size=50, stride=25, output_root='raw_segments_by_sensor/'):
    """
    For every sbj_<id>.csv in data_dir, run sliding_window_segment_sensor
    on each of the 4 sensors and write separate .npy pairs under output_root/<sensor>/.
    """
    logger.info(f"Starting per-sensor segmentation from {data_dir} → {output_root}")
    for fn in sorted(os.listdir(data_dir)):
        if fn.startswith('sbj_') and fn.endswith('.csv'):
            prefix = fn.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, fn))
            if 'activity' not in df.columns:
                logger.warning(f"Skipping {fn}: missing 'activity'")
                continue

            for sensor, cols in SENSORS.items():
                # Ensure the DataFrame has those 3 columns
                for c in cols:
                    if c not in df.columns:
                        logger.error(f"Missing column {c} in {fn}")
                        raise KeyError(f"Column {c} not found in {fn}")

                X, y = sliding_window_segment_sensor(df, cols, window_size, stride, label_col='activity')
                if X.size == 0:
                    logger.warning(f"No windows for {sensor}, subject {prefix}")
                    continue

                out_dir = os.path.join(output_root, sensor)
                save_sensor_segments(X, y, out_dir, sensor, prefix)
        else:
            logger.debug(f"Skipping non-subject file: {fn}")
    logger.info("Per-sensor segmentation complete.")