import torch
import torch.nn as nn

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
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = x.permute(0, 2, 1)
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
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        logits = self.classifier(x)
        return logits
