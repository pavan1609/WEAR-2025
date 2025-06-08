import os
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Label mapping
label_dict = {
    'null': 0,
    'jogging': 1,
    'jogging (rotating arms)': 2,
    'jogging (skipping)': 3,
    'jogging (sidesteps)': 4,
    'jogging (butt-kicks)': 5,
    'stretching (triceps)': 6,
    'stretching (lunging)': 7,
    'stretching (shoulders)': 8,
    'stretching (hamstrings)': 9,
    'stretching (lumbar rotation)': 10,
    'push-ups': 11,
    'push-ups (complex)': 12,
    'sit-ups': 13,
    'sit-ups (complex)': 14,
    'burpees': 15,
    'lunges': 16,
    'lunges (complex)': 17,
    'bench-dips': 18
}

def load_all_subjects(data_dir, sampling_rate=50, save_processed=False, save_path='processed_data.csv'):
    """
    Loads all sbj_*.csv files from a directory, replaces labels with numeric codes,
    adds synthetic timestamps, and concatenates into a single DataFrame.

    Args:
        data_dir (str): Path to directory containing sbj_*.csv files.
        sampling_rate (int): Sampling rate in Hz to compute timestamps.
        save_processed (bool): Whether to save the concatenated DataFrame to CSV.
        save_path (str): Path to save the processed CSV if save_processed=True.

    Returns:
        pd.DataFrame: The concatenated DataFrame of all subjects.
    """
    all_data = []
    time_interval = 1.0 / sampling_rate

    for file in sorted(os.listdir(data_dir)):
        if file.startswith('sbj_') and file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            logger.info(f"Processing file: {file_path}")
            data = pd.read_csv(file_path)

            # Replace labels with numbers
            if 'label' in data.columns:
                data['activity'] = data['label'].replace(label_dict)
                data.drop(columns=['label'], inplace=True)
            elif 'activity' in data.columns:
                data['activity'] = data['activity'].replace(label_dict)
            else:
                logger.error(f"No 'label' or 'activity' column in {file_path}")
                raise ValueError(f"Missing label column in {file}")

            # Create synthetic timestamps
            num_samples = len(data)
            data.insert(0, 'timestamp', np.arange(num_samples) * time_interval)

            # Add subject name
            data['subject'] = file.replace('.csv', '')
            all_data.append(data)
        else:
            logger.debug(f"Skipped file: {file}")

    if not all_data:
        logger.error(f"No subject files found in {data_dir}")
        raise FileNotFoundError(f"No sbj_*.csv files in {data_dir}")

    df = pd.concat(all_data, ignore_index=True)

    if save_processed:
        df.to_csv(save_path, index=False)
        logger.info(f"Saved processed data to {save_path}")

    return df

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load and process all subject CSVs into one DataFrame.')
    parser.add_argument('--data_dir', type=str, default='../raw', help='Path to raw sbj_*.csv files')
    parser.add_argument('--sampling_rate', type=int, default=50, help='Sampling rate (Hz) for timestamp generation')
    parser.add_argument('--save_processed', action='store_true', help='Whether to save processed CSV')
    parser.add_argument('--save_path', type=str, default='processed_data.csv', help='Output path for processed CSV')
    args = parser.parse_args()

    df = load_all_subjects(
        data_dir=args.data_dir,
        sampling_rate=args.sampling_rate,
        save_processed=args.save_processed,
        save_path=args.save_path
    )
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
