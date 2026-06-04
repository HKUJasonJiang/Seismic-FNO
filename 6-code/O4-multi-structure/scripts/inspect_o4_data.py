"""Inspect O4 HDF5 files before launching training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
O4_DIR = SCRIPT_DIR.parent
DEFAULT_BASE_DATA = Path(r"D:\BaiduNetdiskDownload\SesimicTransformerData")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_dir", default=str(DEFAULT_BASE_DATA))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--gm_path", default=None)
    args = parser.parse_args()

    base = Path(args.base_data_dir)
    data_dir = Path(args.data_dir) if args.data_dir else base / "MDOF" / "knet-250" / "Data"
    gm_path = Path(args.gm_path) if args.gm_path else base / "MDOF" / "All_GMs" / "GMs_knet_3474_AF_57.h5"
    structures = json.loads((O4_DIR / "configs" / "o4_structures.json").read_text(encoding="utf-8"))

    print(f"GM: {gm_path} exists={gm_path.exists()}")
    if gm_path.exists():
        with h5py.File(gm_path, "r") as f:
            print(f"  Acc_GMs: shape={f['Acc_GMs'].shape}, dtype={f['Acc_GMs'].dtype}")

    for sid, meta in structures.items():
        path = data_dir / meta["filename"]
        print(f"\n{sid}: {path} exists={path.exists()}")
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            print(f"  Acc_Floor_Response: shape={f['Acc_Floor_Response'].shape}, dtype={f['Acc_Floor_Response'].dtype}")
            print(f"  Blg_Damage_State: shape={f['Blg_Damage_State'].shape}, dtype={f['Blg_Damage_State'].dtype}")
            attrs = {}
            for key, ds in f["Blg_Attributes"].items():
                value = ds[()]
                attrs[key] = value.tolist() if hasattr(value, "tolist") else value
            print(f"  Blg_Attributes: {attrs}")


if __name__ == "__main__":
    main()
