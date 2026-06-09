"""Prepare GM text files for O3 response-spectrum OOD analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
O3_DIR = SCRIPT_DIR.parent
CODE_DIR = O3_DIR.parent
UTILS_DIR = CODE_DIR / "shared" / "utils"
if UTILS_DIR.exists():
    sys.path.insert(0, str(UTILS_DIR))

from splits import make_fixed_474_split  # noqa: E402


CLASSICAL_FILES = [
    "1940-ElCentro-60s-50Hz-ms2.txt",
    "1971-Sanfernando-60s-50Hz-ms2.txt",
    "1985-Michoacan-60s-200Hz-ms2.txt",
    "1991-Kobe-60s-100Hz-ms2.txt",
]


def write_two_column_txt(path: Path, acceleration: np.ndarray, dt: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{len(acceleration)}\n")
        for i, acc in enumerate(acceleration):
            f.write(f"{i * dt:.10g}\t{float(acc):.10g}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare O3 GM inputs for C++ response spectra.")
    parser.add_argument("--gm_h5", default="/data/home/jason/data/SesimicTransformerData/MDOF/All_GMs/GMs_knet_3474_AF_57.h5")
    parser.add_argument("--classical_dir", default=str(Path("data") / "4_ClassicalGMs"))
    parser.add_argument("--output_dir", default=str(O3_DIR / "results" / "gm_ood" / "inputs_scale09"))
    parser.add_argument("--scale_id", type=int, default=9, help="Representative K-NET scale index. 9 is about 1 m/s^2.")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    gm_dir = output_dir / "GMs"
    gm_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "gm_manifest.csv"
    scale_map_path = output_dir / "scale_pga_mapping.csv"

    split = make_fixed_474_split(seed=args.seed)
    train_set = set(int(v) for v in np.concatenate([split.train_gms, split.val_gms]))
    test_set = set(int(v) for v in split.test_gms)

    rows: list[dict[str, object]] = []
    with h5py.File(args.gm_h5, "r") as f:
        acc = f["Acc_GMs"]
        total_scaled, n_steps = acc.shape
        total_gms = total_scaled // 57

        pga_by_scale = []
        sample_gms = np.arange(total_gms)
        for scale_id in range(57):
            idx = sample_gms * 57 + scale_id
            pgas = np.max(np.abs(acc[idx]), axis=1)
            pga_by_scale.append(
                {
                    "scale_id": scale_id,
                    "mean_pga_ms2": float(np.mean(pgas)),
                    "median_pga_ms2": float(np.median(pgas)),
                    "min_pga_ms2": float(np.min(pgas)),
                    "max_pga_ms2": float(np.max(pgas)),
                }
            )

        for gm_id in range(total_gms):
            global_index = gm_id * 57 + args.scale_id
            waveform = acc[global_index]
            group = "train_pool" if gm_id in train_set else "fixed_test" if gm_id in test_set else "unused"
            filename = f"knet_gm{gm_id:04d}_s{args.scale_id:02d}.txt"
            write_two_column_txt(gm_dir / filename, waveform, args.dt)
            rows.append(
                {
                    "name": Path(filename).stem,
                    "filename": filename,
                    "group": group,
                    "source": "knet",
                    "gm_id": gm_id,
                    "scale_id": args.scale_id,
                    "global_index": global_index,
                    "n_steps": n_steps,
                    "dt": args.dt,
                    "pga_ms2": float(np.max(np.abs(waveform))),
                }
            )

    classical_dir = Path(args.classical_dir)
    for filename in CLASSICAL_FILES:
        source = classical_dir / filename
        if not source.exists():
            print(f"[WARN] Missing classical GM: {source}")
            continue
        target = gm_dir / filename
        shutil.copy2(source, target)
        data = np.loadtxt(source, skiprows=1)
        rows.append(
            {
                "name": source.stem,
                "filename": filename,
                "group": "classical",
                "source": "classical",
                "gm_id": "",
                "scale_id": "",
                "global_index": "",
                "n_steps": data.shape[0],
                "dt": float(np.median(np.diff(data[:, 0]))) if data.shape[0] > 1 else "",
                "pga_ms2": float(np.max(np.abs(data[:, 1]))),
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with scale_map_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pga_by_scale[0].keys()))
        writer.writeheader()
        writer.writerows(pga_by_scale)

    print(f"Wrote {len(rows)} GM files to {gm_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Scale PGA mapping: {scale_map_path}")


if __name__ == "__main__":
    main()
