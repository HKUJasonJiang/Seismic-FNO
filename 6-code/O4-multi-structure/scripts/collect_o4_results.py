"""Compile O4 structure summaries into CSV and markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    o4_dir = script_dir.parent
    runs_dir = Path(args.runs_dir) if args.runs_dir else o4_dir / "results" / "runs"
    output_dir = Path(args.output_dir) if args.output_dir else o4_dir / "results" / "compiled"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        test = summary.get("test_metrics", {})
        rows.append(
            {
                "structure_id": summary.get("structure_id"),
                "paper_label": summary.get("paper_label"),
                "type": summary.get("type"),
                "stories": summary.get("stories"),
                "height_m": summary.get("height_m"),
                "best_epoch": summary.get("best_epoch"),
                "best_val_mse": summary.get("best_val_mse"),
                "test_mse": test.get("mse"),
                "test_rmse": test.get("rmse"),
                "test_mae": test.get("mae"),
                "test_r2": test.get("r2"),
                "test_pfa_rmse_g": test.get("pfa_rmse_g"),
                "test_pfa_mae_g": test.get("pfa_mae_g"),
                "test_pfa_r2": test.get("pfa_r2"),
                "train_time_s": summary.get("train_time_s"),
                "peak_gpu_gb": summary.get("peak_gpu_gb"),
                "best_checkpoint": summary.get("best_checkpoint"),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No summary.json files found under {runs_dir}")

    csv_path = output_dir / "o4_multistructure_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "o4_multistructure_table.md"
    lines = [
        "| Structure | Type | Stories | Test R2 | Test RMSE | PFA RMSE (g) | Peak GPU (GB) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {paper_label} | {type} | {stories} | {test_r2:.4f} | {test_rmse:.6f} | {test_pfa_rmse_g:.4f} | {peak_gpu_gb:.2f} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
