import os
from data_loading import load_all_subjects

if __name__ == '__main__':
    RAW_DATA_DIR = os.path.join('..', '/Users/pavan/Downloads/2nd-wear-dataset-challenge/train')
    SAVE_PATH = os.path.join('..', 'processed_data.csv')

    print(f"Loading data from {RAW_DATA_DIR}")
    df = load_all_subjects(data_dir=RAW_DATA_DIR, save_processed=True, save_path=SAVE_PATH)

    print(f"Loaded data shape: {df.shape}")
    print(f"Saved processed data to: {SAVE_PATH}")