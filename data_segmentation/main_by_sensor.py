import os
import pandas as pd
import logging
from data_segmentation.segment_by_sensor import segment_all_subjects_by_sensor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def split_by_subject_acc(processed_csv, out_dir):
    """
    Splits processed_data_acc.csv (with sbj_id, 12 accel cols, activity)
    by sbj_id into CSVs: out_dir/sbj_<id>.csv
    """
    logger.info(f"Splitting {processed_csv} by subject into {out_dir}")
    df = pd.read_csv(processed_csv)
    os.makedirs(out_dir, exist_ok=True)
    for subject in df['sbj_id'].unique():
        subject_df = df[df['sbj_id'] == subject]
        subject_path = os.path.join(out_dir, f"sbj_{subject}.csv")
        subject_df.to_csv(subject_path, index=False)
        logger.info(f"Wrote {len(subject_df)} rows for sbj_{subject} → {subject_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-sensor segmentation pipeline")
    parser.add_argument('--processed_csv', required=True,
                        help="Path to processed_data_acc.csv")
    parser.add_argument('--temp_dir', required=True,
                        help="Temp dir for per-subject CSVs (e.g. raw_data_by_subject_acc)")
    parser.add_argument('--output_root', required=True,
                        help="Root dir for per-sensor segments (e.g. raw_segments_by_sensor)")
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--stride', type=int, default=25)
    args = parser.parse_args()

    split_by_subject_acc(args.processed_csv, args.temp_dir)
    segment_all_subjects_by_sensor(
        data_dir=args.temp_dir,
        window_size=args.window_size,
        stride=args.stride,
        output_root=args.output_root
    )

if __name__ == '__main__':
    main()
