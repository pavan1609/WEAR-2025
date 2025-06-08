import os
import numpy as np
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def check_file(path: str):
    """
    Load a .npy file and check for NaNs or Infs, returning stats.
    """
    try:
        data = np.load(path)
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        stats = {
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
            "mean": float(np.nanmean(data))
        }
        return has_nan, has_inf, stats
    except Exception as e:
        return True, True, {"error": str(e)}


def check_all_npy(data_dirs):
    """
    Iterate over list of directories and check all .npy files.
    """
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            logger.warning(f"Directory does not exist: {data_dir}")
            continue
        logger.info(f"Checking .npy files in: {data_dir}")
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith('.npy'):
                full_path = os.path.join(data_dir, fname)
                has_nan, has_inf, stats = check_file(full_path)
                if has_nan or has_inf:
                    logger.error(f"Corrupted: {fname} — NaN: {has_nan}, Inf: {has_inf}")
                    if 'error' in stats:
                        logger.error(f"Error details: {stats['error']}")
                else:
                    logger.info(f"OK: {fname} — Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sanity check .npy files in one or more directories')
    parser.add_argument('dirs', nargs='+', help='One or more directories to scan for .npy files')
    args = parser.parse_args()

    check_all_npy(args.dirs)
