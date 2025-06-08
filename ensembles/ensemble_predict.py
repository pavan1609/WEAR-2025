import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import ast
import argparse
from torch.utils.data import Dataset, DataLoader

# ─── PatchTST Definition ───────────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    def __init__(self, input_channels, patch_size, d_model):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)       # (batch, channels, seq_len)
        x = self.proj(x)            # (batch, d_model, n_patches)
        x = x.permute(0, 2, 1)       # (batch, n_patches, d_model)
        x = self.norm(x)
        return x

class PatchTSTModel(nn.Module):
    def __init__(
        self,
        input_channels,
        patch_size,
        seq_len,
        d_model,
        n_heads,
        num_layers,
        num_classes,
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(input_channels, patch_size, d_model)
        n_patches = seq_len // patch_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model * n_patches, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        x = self.patch_embed(x)      # (batch, n_patches, d_model)
        x = self.transformer(x)      # (batch, n_patches, d_model)
        x = x.flatten(start_dim=1)   # (batch, n_patches * d_model)
        logits = self.classifier(x)  # (batch, num_classes)
        return logits

# ─── Dataset for PatchTST Inference ────────────────────────────────────────────
class TestSensorDataset(Dataset):
    """
    Given a list of (50×3) numpy windows, returns normalized torch tensors.
    """
    def __init__(self, data_list):
        self.data_list = data_list  # list of np arrays shape (50,3)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = torch.tensor(self.data_list[idx], dtype=torch.float32)  # (50,3)
        # normalize per window
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
        x = (x - mean) / std
        return x  # shape: (50,3)

# ─── Main Ensemble Function ────────────────────────────────────────────────────
def ensemble_predict(
    test_csv,
    classical_models_dir,
    patchtst_models_dir,
    output_csv,
    patch_size=10,
    d_model=64,
    n_heads=4,
    num_layers=3,
    dropout=0.1
):
    """
    1. Read test.csv (contains id, sensor_location, x_axis, y_axis, z_axis).
    2. For each sensor window, compute:
       a) Classical LightGBM probability (load from classical_models_dir/lgbm_<sensor>.pkl).
       b) PatchTST probability (load from patchtst_models_dir/patchtst_<sensor>.pth).
    3. Average the two probability vectors (0.5 * classical + 0.5 * patchtst).
    4. argmax → final label. Write CSV {id, target_feature}.
    """
    # 1) Load test DataFrame
    try:
        df = pd.read_csv(test_csv)
    except Exception as e:
        print(f"Error reading test CSV at {test_csv}: {e}")
        return

    if 'id' not in df.columns or 'sensor_location' not in df.columns \
       or 'x_axis' not in df.columns or 'y_axis' not in df.columns or 'z_axis' not in df.columns:
        print("Error: test CSV must contain columns: id, sensor_location, x_axis, y_axis, z_axis")
        return

    ids = df['id'].tolist()
    sensors = df['sensor_location'].tolist()

    # 2) Load classical LightGBM models
    sensors_list = ['right_arm', 'right_leg', 'left_leg', 'left_arm']
    classical_boosters = {}
    for sensor in sensors_list:
        model_path = os.path.join(classical_models_dir, f"lgbm_{sensor}.pkl")
        if not os.path.exists(model_path):
            print(f"Error: Classical model not found for {sensor} at {model_path}")
            return
        with open(model_path, "rb") as f:
            classical_boosters[sensor] = pickle.load(f)

    # 3) Load PatchTST models (per sensor)
    patchtst_models = {}
    for sensor in sensors_list:
        cp_path = os.path.join(patchtst_models_dir, f"patchtst_{sensor}.pth")
        if not os.path.exists(cp_path):
            print(f"Error: PatchTST checkpoint not found for {sensor} at {cp_path}")
            return

        checkpoint = torch.load(cp_path, map_location='cpu')

        # Build model architecture exactly as in training
        model = PatchTSTModel(
            input_channels=3,
            patch_size=patch_size,
            seq_len=50,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            num_classes=19,
            dropout=dropout
        )
        try:
            model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading state_dict for PatchTST model {sensor}: {e}")
            return
        model.eval()
        patchtst_models[sensor] = model

    # 4) Group windows by sensor for batch inference
    sensor_data = {sensor: [] for sensor in sensors_list}
    idx_map = {sensor: [] for sensor in sensors_list}
    for idx, row in df.iterrows():
        sensor = row.sensor_location
        if sensor not in sensors_list:
            print(f"Warning: Unknown sensor_location '{sensor}' for id {row.id}; skipping")
            continue
        try:
            x = np.array(ast.literal_eval(row.x_axis), dtype=np.float32)
            y = np.array(ast.literal_eval(row.y_axis), dtype=np.float32)
            z = np.array(ast.literal_eval(row.z_axis), dtype=np.float32)
        except Exception as e:
            print(f"Error parsing axes for row {idx}: {e}")
            # Fallback to zeros
            x = np.zeros(50, dtype=np.float32)
            y = np.zeros(50, dtype=np.float32)
            z = np.zeros(50, dtype=np.float32)

        if x.shape[0] != 50 or y.shape[0] != 50 or z.shape[0] != 50:
            print(f"Warning: axis length != 50 for id {row.id}, sensor {sensor}. Filling with zeros")
            x = np.zeros(50, dtype=np.float32)
            y = np.zeros(50, dtype=np.float32)
            z = np.zeros(50, dtype=np.float32)

        window = np.stack([x, y, z], axis=1)  # shape (50,3)
        sensor_data[sensor].append(window)
        idx_map[sensor].append(idx)

    # Prepare array to hold combined probabilities
    num_rows = len(df)
    all_probs = np.zeros((num_rows, 19), dtype=np.float32)

    # 5) For each sensor, predict classical + patchtst, then average
    for sensor in sensors_list:
        indices = idx_map[sensor]
        windows = sensor_data[sensor]
        n = len(windows)
        if n == 0:
            print(f"No test windows for sensor {sensor}; skipping")
            continue

        # 5a) Classical: compute static features for each window
        features = []
        for W in windows:
            # Basic stats
            mean = W.mean(axis=0)   # (3,)
            std  = W.std(axis=0)    # (3,)
            mn   = W.min(axis=0)    # (3,)
            mx   = W.max(axis=0)    # (3,)
            # Magnitude stats
            mag = np.linalg.norm(W, axis=1)  # (50,)
            mag_feats = np.array([mag.mean(), mag.std(), mag.min(), mag.max()])
            feat = np.concatenate([mean, std, mn, mx, mag_feats], axis=0)  # (16,)
            features.append(feat)
        X_feats = np.stack(features, axis=0)  # shape (n,16)

        booster = classical_boosters[sensor]
        try:
            class_prob = booster.predict(X_feats)  # shape (n,19)
        except Exception as e:
            print(f"Error predicting classical for {sensor}: {e}")
            class_prob = np.zeros((n, 19), dtype=np.float32)

        # 5b) PatchTST: build DataLoader for windows
        dataset = TestSensorDataset(windows)
        loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        model   = patchtst_models[sensor]
        patch_probs_list = []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch)  # (B,19)
                prob = nn.functional.softmax(logits, dim=1).cpu().numpy()
                patch_probs_list.append(prob)
        patch_probs = np.concatenate(patch_probs_list, axis=0)  # (n,19)

        # 5c) Average classical & patchtst probabilities
        combined = 0.5 * class_prob + 0.5 * patch_probs
        for i, idx in enumerate(indices):
            all_probs[idx] = combined[i]

    # 6) Final labels = argmax
    final_labels = np.argmax(all_probs, axis=1).astype(int)
    submission = pd.DataFrame({'id': ids, 'target_feature': final_labels})
    submission.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} with {len(ids)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble GBDT + PatchTST predictions")
    parser.add_argument("--test_csv",    required=True,
                        help="Path to Kaggle test.csv")
    parser.add_argument("--classical_models_dir",  required=True,
                        help="Directory where lgbm_<sensor>.pkl files are stored")
    parser.add_argument("--patchtst_models_dir",  required=True,
                        help="Directory where patchtst_<sensor>.pth files are stored")
    parser.add_argument("--output_csv",  default="submissions/submission_final.csv",
                        help="Filename for the final submission CSV")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    ensemble_predict(
        args.test_csv,
        args.classical_models_dir,
        args.patchtst_models_dir,
        args.output_csv
    )
