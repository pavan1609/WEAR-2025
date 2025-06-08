# File: baseline/patchtst/train_patchtst.py

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from tqdm import tqdm

# Import the PatchTSTModel from the same directory
from baseline.patchtst.patchtst_model import PatchTSTModel


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SensorWindowDataset(Dataset):
    def __init__(self, X, y):
        # X: numpy array of shape (N, 50, 3), y: numpy array of shape (N,)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # shape: (50, 3)
        # Normalize per window
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
        x = (x - mean) / std
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def load_sensor_data(sensor_dir):
    """
    Loads all *_X.npy and *_y.npy pairs under sensor_dir,
    but discards any window that is NaN/Inf or zero‐variance.
    Returns X_all (N,50,3) and y_all (N,).
    """
    X_list, y_list = [], []
    for x_path in sorted(glob.glob(os.path.join(sensor_dir, "*_X.npy"))):
        prefix = x_path[:-6]
        y_path = prefix + '_y.npy'
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Missing {y_path}")
        X = np.load(x_path)
        y = np.load(y_path)

        # Filter: finite values and nonzero variance
        mask_finite = np.all(np.isfinite(X), axis=(1, 2))
        var_per_window = np.var(X, axis=(1, 2))
        mask_var = var_per_window > 0
        good_mask = mask_finite & mask_var
        if not np.all(good_mask):
            print(f"  → Skipping {np.sum(~good_mask)} bad windows in {os.path.basename(x_path)}")
        X_clean = X[good_mask]
        y_clean = y[good_mask]
        if len(X_clean) == 0:
            continue
        X_list.append(X_clean)
        y_list.append(y_clean)
    if len(X_list) == 0:
        raise RuntimeError(f"No valid windows found under {sensor_dir}")
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all


def train_sensor_patchtst(
    sensor,
    segments_root,
    output_dir,
    patch_size,
    d_model,
    n_heads,
    num_layers,
    dropout,
    epochs,
    batch_size,
    lr,
    device
):
    """
    Train a PatchTSTModel on all data for a given sensor and save the checkpoint.
    """
    sensor_dir = os.path.join(segments_root, sensor)
    X_all, y_all = load_sensor_data(sensor_dir)
    print(f"Loaded {sensor}: X={X_all.shape}, y={y_all.shape}")

    dataset = SensorWindowDataset(X_all, y_all)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = PatchTSTModel(
        input_channels=3,
        patch_size=patch_size,
        seq_len=50,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        num_classes=19,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.cpu().numpy())

        avg_loss = total_loss / len(dataset)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        scheduler.step(train_f1)
        print(f"{sensor} Epoch {epoch}/{epochs} — loss: {avg_loss:.4f}, train_macro_F1: {train_f1:.4f}")

        if train_f1 > best_f1:
            best_f1 = train_f1
            ckpt_path = os.path.join(output_dir, f"patchtst_{sensor}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f" ▶ Saved best {sensor} checkpoint (F1={best_f1:.4f}) to {ckpt_path}")

    print(f"Training complete for {sensor}. Best F1: {best_f1:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PatchTST per sensor on full data")
    parser.add_argument("--segments_root", required=True,
                        help="Root directory of per-sensor folders (raw_segments_by_sensor)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save patchtst_<sensor>.pth checkpoints")
    parser.add_argument("--sensors", type=str, default="right_arm,right_leg,left_leg,left_arm",
                        help="Comma-separated list of sensors to train (default: all four)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    sensors_to_train = args.sensors.split(",")
    for sensor in sensors_to_train:
        sensor = sensor.strip()
        if sensor == "":
            continue
        train_sensor_patchtst(
            sensor=sensor,
            segments_root=args.segments_root,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device
        )