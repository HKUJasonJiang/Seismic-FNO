"""Streaming metrics for regression without retaining all predictions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class RegressionMeter:
    n: int = 0
    sum_sq_error: float = 0.0
    sum_abs_error: float = 0.0
    sum_y: float = 0.0
    sum_y2: float = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach()
        target = target.detach()
        diff = pred - target
        self.n += target.numel()
        self.sum_sq_error += float(torch.sum(diff * diff).item())
        self.sum_abs_error += float(torch.sum(torch.abs(diff)).item())
        self.sum_y += float(torch.sum(target).item())
        self.sum_y2 += float(torch.sum(target * target).item())

    def compute(self) -> dict[str, float]:
        if self.n == 0:
            return {"mse": math.nan, "rmse": math.nan, "mae": math.nan, "r2": math.nan}
        mse = self.sum_sq_error / self.n
        mae = self.sum_abs_error / self.n
        sst = self.sum_y2 - (self.sum_y * self.sum_y / self.n)
        r2 = 1.0 - self.sum_sq_error / sst if sst > 0 else math.nan
        return {"mse": mse, "rmse": math.sqrt(mse), "mae": mae, "r2": r2}


@dataclass
class PFAMeter:
    n: int = 0
    sum_sq_error: float = 0.0
    sum_abs_error: float = 0.0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    g: float = 9.81

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_pfa = torch.amax(torch.abs(pred.detach()), dim=-1) / self.g
        target_pfa = torch.amax(torch.abs(target.detach()), dim=-1) / self.g
        diff = pred_pfa - target_pfa
        self.n += target_pfa.numel()
        self.sum_sq_error += float(torch.sum(diff * diff).item())
        self.sum_abs_error += float(torch.sum(torch.abs(diff)).item())
        self.sum_y += float(torch.sum(target_pfa).item())
        self.sum_y2 += float(torch.sum(target_pfa * target_pfa).item())

    def compute(self) -> dict[str, float]:
        if self.n == 0:
            return {"pfa_mse_g2": math.nan, "pfa_rmse_g": math.nan, "pfa_mae_g": math.nan, "pfa_r2": math.nan}
        mse = self.sum_sq_error / self.n
        sst = self.sum_y2 - (self.sum_y * self.sum_y / self.n)
        r2 = 1.0 - self.sum_sq_error / sst if sst > 0 else math.nan
        return {
            "pfa_mse_g2": mse,
            "pfa_rmse_g": math.sqrt(mse),
            "pfa_mae_g": self.sum_abs_error / self.n,
            "pfa_r2": r2,
        }

