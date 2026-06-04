"""Collect O1 training/evaluation metadata into plotting-friendly files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


O1_DIR = Path(__file__).resolve().parent.parent


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_training_runs(run_root: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(run_root.glob("*/summary.json")):
        data = read_json(summary_path)
        if not data:
            continue
        config = read_json(summary_path.parent / "config.json") or {}
        rows.append(
            {
                "model": data.get("model", summary_path.parent.name),
                "run_dir": str(summary_path.parent),
                "best_checkpoint": data.get("best_checkpoint", ""),
                "best_val_mse": data.get("best_val_mse", ""),
                "train_time_s": data.get("train_time_s", ""),
                "train_time_h": data.get("train_time_s", 0) / 3600 if data.get("train_time_s") else "",
                "peak_gpu_gb": data.get("peak_gpu_gb", ""),
                "params_numel": (data.get("params") or {}).get("numel", ""),
                "params_m": (data.get("params") or {}).get("numel_m", ""),
                "batch_size": config.get("batch_size", ""),
                "epochs": config.get("epochs", ""),
                "split": config.get("split", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect O1 run metadata.")
    parser.add_argument("--run_root", default=str(O1_DIR / "results" / "runs"))
    parser.add_argument("--evaluation_csv", default=str(O1_DIR / "results" / "evaluation" / "table4_metrics.csv"))
    parser.add_argument("--output_dir", default=str(O1_DIR / "results" / "compiled"))
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_rows = collect_training_runs(Path(args.run_root))
    write_csv(output / "training_runs.csv", train_rows)
    (output / "training_runs.json").write_text(json.dumps(train_rows, indent=2), encoding="utf-8")

    eval_path = Path(args.evaluation_csv)
    if eval_path.exists():
        (output / "table4_metrics.csv").write_text(eval_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "training_runs_csv": str(output / "training_runs.csv"),
        "training_runs_json": str(output / "training_runs.json"),
        "table4_metrics_csv": str(output / "table4_metrics.csv") if eval_path.exists() else "",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Compiled O1 results written to {output}")


if __name__ == "__main__":
    main()

