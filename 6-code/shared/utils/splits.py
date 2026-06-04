"""Shared split utilities for SeFNO ES experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitBundle:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_gms: np.ndarray
    val_gms: np.ndarray
    test_gms: np.ndarray
    scale_indices: np.ndarray


def make_fixed_474_split(
    total_gms: int = 3474,
    total_scales: int = 57,
    test_size: int = 474,
    train_gms_used: int = 3000,
    use_scales_count: int = 57,
    seed: int = 42,
) -> SplitBundle:
    """Return the canonical FNO-Large split.

    Canonical protocol:
    1. Shuffle 3474 original ground motions with seed 42.
    2. Use the last 474 GMs as the fixed test set.
    3. Use the first `train_gms_used` GMs from the remaining pool.
    4. Split that pool into 80/20 train/validation.
    5. Keep every scale of a GM inside the same split.
    """
    rng = np.random.RandomState(seed)
    shuffled_gms = rng.permutation(total_gms)
    test_gms = shuffled_gms[-test_size:]
    pool_gms = shuffled_gms[:-test_size]

    if train_gms_used > len(pool_gms):
        raise ValueError(f"train_gms_used={train_gms_used} exceeds pool size={len(pool_gms)}")

    selected_pool_gms = pool_gms[:train_gms_used]
    train_gms, val_gms = train_test_split(selected_pool_gms, test_size=0.2, random_state=seed)

    if use_scales_count >= total_scales:
        scale_indices = np.arange(total_scales)
    else:
        scale_indices = np.linspace(0, total_scales - 1, use_scales_count)
        scale_indices = np.unique(np.round(scale_indices).astype(int))

    def flatten(gms: np.ndarray, scales: np.ndarray) -> np.ndarray:
        gm_grid, scale_grid = np.meshgrid(gms, scales, indexing="ij")
        return (gm_grid * total_scales + scale_grid).reshape(-1)

    all_scales = np.arange(total_scales)
    return SplitBundle(
        train_indices=flatten(train_gms, scale_indices),
        val_indices=flatten(val_gms, scale_indices),
        test_indices=flatten(test_gms, all_scales),
        train_gms=train_gms,
        val_gms=val_gms,
        test_gms=test_gms,
        scale_indices=scale_indices,
    )


def save_split(bundle: SplitBundle, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dataset_indices.pkl").open("wb") as f:
        pickle.dump(
            {
                "train_indices": bundle.train_indices,
                "val_indices": bundle.val_indices,
                "test_indices": bundle.test_indices,
                "train_gms": bundle.train_gms,
                "val_gms": bundle.val_gms,
                "test_gms": bundle.test_gms,
                "scale_indices": bundle.scale_indices,
            },
            f,
        )

    (output / "split_info.txt").write_text(
        "\n".join(
            [
                "Canonical split: fixed 474-GM test + 3000-GM pool + 80/20 train/val",
                f"Train GMs: {len(bundle.train_gms)}",
                f"Val GMs: {len(bundle.val_gms)}",
                f"Test GMs: {len(bundle.test_gms)}",
                f"Train samples: {len(bundle.train_indices)}",
                f"Val samples: {len(bundle.val_indices)}",
                f"Test samples: {len(bundle.test_indices)}",
                f"Scale count: {len(bundle.scale_indices)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

