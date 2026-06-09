"""Baseline neural surrogates for O1 full-data comparison.

All models use the same interface:
    input:  (batch, 1, 3000)
    output: (batch, 1, 3000)

The set is intentionally broad but still focused:
- MLP: global fully connected sequence mapping
- BiLSTM: recurrent temporal baseline
- ResCNN1D: dilated residual CNN / TCN-style baseline
- UNet1D: multi-scale convolutional encoder-decoder
- PatchTransformer: attention-based sequence baseline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class ResCNN1D(nn.Module):
    """Dilated residual CNN / TCN-style baseline."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 256,
        kernel_size: int = 7,
        n_blocks: int = 8,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.lift = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        dilation_cycle = [1, 2, 4, 8, 16]
        self.blocks = nn.ModuleList(
            ResBlock1D(channels, kernel_size, dilation_cycle[i % len(dilation_cycle)])
            for i in range(n_blocks)
        )
        self.proj = nn.Conv1d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.proj(x)


class BiLSTMModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_size: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_size * 2, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.head(x).permute(0, 2, 1)


class MLPModel(nn.Module):
    def __init__(
        self,
        seq_len: int = 3072,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden: list[int] | None = None,
    ):
        super().__init__()
        hidden = hidden or [1024, 1024, 1024, 1024]
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        dims = [seq_len * in_channels, *hidden, seq_len * out_channels]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _channels, time = x.shape
        if time < self.seq_len:
            x = F.pad(x, (0, self.seq_len - time))
        else:
            x = x[:, :, : self.seq_len]
        x = self.net(x.reshape(batch, -1))
        return x.reshape(batch, self.out_channels, self.seq_len)[:, :, :time]


class DoubleConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        enc_channels: list[int] | None = None,
        pad_to: int = 3008,
    ):
        super().__init__()
        enc_channels = enc_channels or [64, 128, 256, 512]
        self.pad_to = pad_to
        self.enc_convs = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = in_channels
        for next_channels in enc_channels:
            self.enc_convs.append(DoubleConv1D(channels, next_channels))
            self.downs.append(nn.Conv1d(next_channels, next_channels, 2, stride=2))
            channels = next_channels
        self.bottleneck = DoubleConv1D(enc_channels[-1], enc_channels[-1])
        self.ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        channels = enc_channels[-1]
        for skip_channels in reversed(enc_channels):
            self.ups.append(nn.ConvTranspose1d(channels, skip_channels, 2, stride=2))
            self.dec_convs.append(DoubleConv1D(skip_channels * 2, skip_channels))
            channels = skip_channels
        self.proj = nn.Conv1d(enc_channels[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _batch, _channels, time = x.shape
        if time < self.pad_to:
            x = F.pad(x, (0, self.pad_to - time))
        skips = []
        for enc, down in zip(self.enc_convs, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.dec_convs, reversed(skips)):
            x = up(x)
            if x.shape[-1] < skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            elif x.shape[-1] > skip.shape[-1]:
                x = x[:, :, : skip.shape[-1]]
            x = dec(torch.cat([x, skip], dim=1))
        return self.proj(x)[:, :, :time]


class PatchTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        patch_size: int = 15,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 10,
        seq_len: int = 3000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.patch_embed = nn.Linear(patch_size * in_channels, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, patch_size * out_channels)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time = x.shape
        pad_len = (self.patch_size - time % self.patch_size) % self.patch_size
        if pad_len:
            x = F.pad(x, (0, pad_len))
        padded_time = time + pad_len
        n_patches = padded_time // self.patch_size
        x = x.reshape(batch, channels, n_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(batch, n_patches, channels * self.patch_size)
        x = self.input_norm(self.patch_embed(x)) + self.pos_embed[:, :n_patches, :]
        x = self.encoder(x)
        x = self.output_proj(self.output_norm(x))
        x = x.reshape(batch, n_patches, self.out_channels, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(batch, self.out_channels, padded_time)
        return x[:, :, :time]


@dataclass(frozen=True)
class ModelSpec:
    cls: type[nn.Module]
    config: dict[str, Any]
    paper_label: str


BASELINE_REGISTRY: dict[str, ModelSpec] = {
    "MLP": ModelSpec(
        MLPModel,
        {"seq_len": 3072, "in_channels": 1, "out_channels": 1, "hidden": [1024, 1024, 1024, 1024]},
        "MLP",
    ),
    "BiLSTM": ModelSpec(
        BiLSTMModel,
        {"in_channels": 1, "out_channels": 1, "hidden_size": 512, "num_layers": 2},
        "BiLSTM",
    ),
    "ResCNN1D": ModelSpec(
        ResCNN1D,
        {"in_channels": 1, "out_channels": 1, "channels": 256, "kernel_size": 7, "n_blocks": 8},
        "Dilated 1D-CNN",
    ),
    "UNet1D": ModelSpec(
        UNet1D,
        {"in_channels": 1, "out_channels": 1, "enc_channels": [64, 128, 256, 512]},
        "1D U-Net",
    ),
    "Transformer": ModelSpec(
        PatchTransformer,
        {
            "in_channels": 1,
            "out_channels": 1,
            "patch_size": 15,
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 10,
            "seq_len": 3000,
        },
        "Patch Transformer",
    ),
}


def build_model(name: str, **overrides: Any) -> nn.Module:
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown model {name}. Options: {list(BASELINE_REGISTRY)}")
    spec = BASELINE_REGISTRY[name]
    config = {**spec.config, **overrides}
    return spec.cls(**config)


def count_parameters(model: nn.Module) -> dict[str, float | int]:
    numel = sum(param.numel() for param in model.parameters())
    real_params = sum(param.numel() * 2 if param.is_complex() else param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "numel": numel,
        "numel_m": numel / 1e6,
        "real_params": real_params,
        "real_params_m": real_params / 1e6,
        "trainable": trainable,
    }


def verify_forward(seq_len: int = 3000) -> list[dict[str, Any]]:
    x = torch.randn(2, 1, seq_len)
    rows = []
    for name in BASELINE_REGISTRY:
        model = build_model(name).eval()
        with torch.no_grad():
            y = model(x)
        stats = count_parameters(model)
        rows.append({"model": name, "output_shape": list(y.shape), **stats})
    return rows


if __name__ == "__main__":
    for row in verify_forward():
        print(
            f"{row['model']:>12s} | params={row['numel_m']:.2f}M | "
            f"real={row['real_params_m']:.2f}M | output={row['output_shape']}"
        )

