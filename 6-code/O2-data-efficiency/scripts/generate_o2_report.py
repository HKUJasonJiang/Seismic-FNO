"""Generate O2 data-efficiency markdown, LaTeX source, and PDF report."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
O2_DIR = SCRIPT_DIR.parent

MODEL_ORDER = ["FNO-Large", "Transformer", "BiLSTM"]
RUN_ORDER = [
    "E-Base",
    "E-AC80",
    "E-AC60",
    "E-AC40",
    "E-AC20",
    "E-AC10",
    "E-AC05",
    "E-GM80",
    "E-GM60",
    "E-GM40",
    "E-GM20",
    "E-J80",
    "E-J60",
    "E-J40",
    "E-J20",
]
AXIS_ORDER = ["full", "ac", "gm", "joint"]
MODEL_COLORS = {
    "FNO-Large": "#1f77b4",
    "Transformer": "#d62728",
    "BiLSTM": "#2ca02c",
}
AXIS_MARKERS = {
    "full": "o",
    "ac": "s",
    "gm": "^",
    "joint": "D",
}


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        "train_gms_used",
        "use_scales_count",
        "pool_samples",
        "train_samples",
        "val_samples",
        "test_samples",
        "mse",
        "rmse",
        "mae",
        "r2",
        "pfa_rmse_g",
        "spectral_log_mse",
        "high_freq_log_mse",
        "train_time_s",
        "peak_gpu_gb",
        "best_epoch",
        "r2_retention",
        "mse_ratio",
        "time_ratio",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def sorted_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_model_order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    df["_run_order"] = df["run_id"].map({r: i for i, r in enumerate(RUN_ORDER)}).fillna(99)
    return df.sort_values(["_model_order", "_run_order"]).drop(columns=["_model_order", "_run_order"])


def format_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def format_samples(value: float) -> str:
    if pd.isna(value):
        return "-"
    value = float(value)
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    rows = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row.get(col, "")
            if col in {"mse", "r2", "pfa_rmse_g", "r2_retention", "mse_ratio", "time_ratio"}:
                cells.append(format_float(value, 4))
            elif col in {"train_time_s"}:
                cells.append(format_float(value / 3600.0 if not pd.isna(value) else value, 2))
            elif col in {"peak_gpu_gb"}:
                cells.append(format_float(value, 2))
            elif col in {"pool_samples", "train_samples", "val_samples", "test_samples"}:
                cells.append(format_samples(value))
            elif col in {"train_gms_used", "use_scales_count", "best_epoch"} and not pd.isna(value):
                cells.append(str(int(value)))
            else:
                cells.append(str(value))
        rows.append("|" + "|".join(cells) + "|")
    return "\n".join(rows)


def plot_mse_by_axis(df: pd.DataFrame, fig_dir: Path) -> None:
    axes_to_plot = ["ac", "gm", "joint"]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, axis_name in zip(axs, axes_to_plot):
        sub = df[df["axis"] == axis_name]
        for model in MODEL_ORDER:
            mdf = sub[sub["model"] == model].sort_values("pool_samples")
            if mdf.empty:
                continue
            ax.plot(
                mdf["pool_samples"] / 1000.0,
                mdf["mse"],
                marker="o",
                linewidth=2,
                label=model,
                color=MODEL_COLORS.get(model),
            )
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        ax.set_title(axis_name.upper())
        ax.set_xlabel("Pool samples (k)")
    axs[0].set_ylabel("Fixed-test MSE")
    axs[-1].legend(frameon=False, fontsize=9)
    fig.suptitle("O2 Data Efficiency: Test MSE Degradation")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_mse_degradation.png", dpi=220)
    fig.savefig(fig_dir / "fig1_mse_degradation.pdf")
    plt.close(fig)


def plot_retention(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for model in MODEL_ORDER:
        for axis_name in AXIS_ORDER:
            sub = df[(df["model"] == model) & (df["axis"] == axis_name)].sort_values("pool_samples")
            if sub.empty:
                continue
            label = model if axis_name == "joint" else f"{model} ({axis_name})"
            alpha = 0.95 if axis_name == "joint" else 0.35
            ax.plot(
                sub["pool_samples"] / 1000.0,
                sub["r2_retention"],
                marker=AXIS_MARKERS.get(axis_name, "o"),
                linewidth=2 if axis_name == "joint" else 1.3,
                alpha=alpha,
                label=label,
                color=MODEL_COLORS.get(model),
            )
    ax.axhline(0.95, color="#555555", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Pool samples (k)")
    ax.set_ylabel("R2 retention vs. each model's E-Base")
    ax.set_title("Relative Fidelity Retention")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_r2_retention.png", dpi=220)
    fig.savefig(fig_dir / "fig2_r2_retention.pdf")
    plt.close(fig)


def plot_heatmaps(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    gms_vals = [3000, 2400, 1800, 1200, 600]
    scale_vals = [57, 46, 34, 23, 11, 6, 3]
    for ax, model in zip(axs, MODEL_ORDER):
        sub = df[df["model"] == model]
        mat = np.full((len(scale_vals), len(gms_vals)), np.nan)
        for i, scale in enumerate(scale_vals):
            for j, gms in enumerate(gms_vals):
                match = sub[(sub["train_gms_used"] == gms) & (sub["use_scales_count"] == scale)]
                if not match.empty:
                    mat[i, j] = float(match.iloc[0]["r2_retention"])
        im = ax.imshow(mat, vmin=0.75, vmax=1.02, cmap="viridis", aspect="auto")
        ax.set_title(model)
        ax.set_xticks(range(len(gms_vals)), [str(v) for v in gms_vals], rotation=35)
        ax.set_yticks(range(len(scale_vals)), [str(v) for v in scale_vals])
        ax.set_xlabel("Pool GMs")
        for i in range(len(scale_vals)):
            for j in range(len(gms_vals)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    axs[0].set_ylabel("AC scales")
    fig.colorbar(im, ax=axs, shrink=0.85, label="R2 retention")
    fig.suptitle("Retention Heatmap over GM and AC Budgets")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_retention_heatmap.png", dpi=220)
    fig.savefig(fig_dir / "fig3_retention_heatmap.pdf")
    plt.close(fig)


def plot_training_curves(results_root: Path, fig_dir: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    selected = ["E-AC20", "E-GM20", "E-J20"]
    for ax, model in zip(axs, MODEL_ORDER):
        for run_id in selected:
            path = results_root / "runs" / model / run_id / "training_log.csv"
            if not path.exists():
                continue
            log = pd.read_csv(path)
            if log.empty:
                continue
            ax.plot(log["epoch"], log["val_mse"], linewidth=1.8, label=run_id)
        ax.set_title(model)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
    axs[0].set_ylabel("Validation MSE")
    axs[-1].legend(frameon=False, fontsize=9)
    fig.suptitle("Representative Reduced-Data Validation Curves")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_training_curves.png", dpi=220)
    fig.savefig(fig_dir / "fig4_training_curves.pdf")
    plt.close(fig)


def plot_summary_table(df: pd.DataFrame, fig_dir: Path) -> None:
    cols = ["model", "run_id", "train_gms_used", "use_scales_count", "pool_samples", "mse", "r2", "r2_retention"]
    table_df = sorted_df(df)[cols].copy()
    table_df["pool_samples"] = table_df["pool_samples"].map(format_samples)
    for col in ["mse", "r2", "r2_retention"]:
        table_df[col] = table_df[col].map(lambda v: format_float(v, 4))
    for col in ["train_gms_used", "use_scales_count"]:
        table_df[col] = table_df[col].map(lambda v: "-" if pd.isna(v) else str(int(v)))
    fig_h = max(6.0, 0.22 * len(table_df) + 1.0)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=["Model", "Run", "GMs", "ACs", "Pool", "MSE", "R2", "R2 Ret."],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eaeaea")
        else:
            cell.set_edgecolor("#dddddd")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_results_table.png", dpi=220)
    fig.savefig(fig_dir / "fig5_results_table.pdf")
    plt.close(fig)


def build_conclusions(df: pd.DataFrame) -> list[str]:
    conclusions = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        base = sub[sub["run_id"] == "E-Base"]
        if base.empty:
            continue
        base_mse = float(base.iloc[0]["mse"])
        base_r2 = float(base.iloc[0]["r2"])
        reduced = sub[sub["run_id"] != "E-Base"].dropna(subset=["mse", "r2_retention"])
        if reduced.empty:
            conclusions.append(f"{model}: E-Base MSE={base_mse:.4f}, R2={base_r2:.4f}; reduced runs pending.")
            continue
        best_small = reduced.sort_values(["pool_samples", "mse"]).iloc[0]
        robust = reduced[reduced["r2_retention"] >= 0.95].sort_values("pool_samples")
        if robust.empty:
            robust_text = "no reduced configuration retained 95% of E-Base R2"
        else:
            r = robust.iloc[0]
            robust_text = f"{r['run_id']} retained >=95% R2 with {format_samples(r['pool_samples'])} pool samples"
        conclusions.append(
            f"{model}: E-Base MSE={base_mse:.4f}, R2={base_r2:.4f}; "
            f"smallest completed run {best_small['run_id']} has MSE={best_small['mse']:.4f}; {robust_text}."
        )
    return conclusions


def write_markdown(df: pd.DataFrame, report_dir: Path, conclusions: list[str]) -> None:
    report_path = report_dir / "O2_data_efficiency_report.md"
    full_table_cols = [
        "model",
        "run_id",
        "axis",
        "train_gms_used",
        "use_scales_count",
        "pool_samples",
        "mse",
        "r2",
        "pfa_rmse_g",
        "r2_retention",
        "train_time_s",
        "peak_gpu_gb",
    ]
    text = [
        "# O2 Data Efficiency Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Protocol",
        "",
        "The O2 study keeps the fixed 474-GM test set from O1 unchanged and prunes only the supervised train/validation pool. FNO-Large, Transformer, and BiLSTM are trained with the O1-final hyperparameters. E-Base is imported from O1 full-data results.",
        "",
        "## Key Conclusions",
        "",
    ]
    text.extend([f"- {item}" for item in conclusions])
    text.extend(
        [
            "",
            "## Figures",
            "",
            "![Test MSE degradation](figures/fig1_mse_degradation.png)",
            "",
            "![R2 retention](figures/fig2_r2_retention.png)",
            "",
            "![Retention heatmap](figures/fig3_retention_heatmap.png)",
            "",
            "![Training curves](figures/fig4_training_curves.png)",
            "",
            "![Results table](figures/fig5_results_table.png)",
            "",
            "## Quantitative Table",
            "",
            markdown_table(
                sorted_df(df),
                full_table_cols,
                ["Model", "Run", "Axis", "GMs", "ACs", "Pool", "MSE", "R2", "PFA RMSE", "R2 Ret.", "Time(h)", "GPU(GB)"],
            ),
            "",
        ]
    )
    report_path.write_text("\n".join(text), encoding="utf-8")


def write_latex(df: pd.DataFrame, report_dir: Path, conclusions: list[str]) -> None:
    report_path = report_dir / "O2_data_efficiency_report.tex"
    compact = sorted_df(df)[["model", "run_id", "train_gms_used", "use_scales_count", "pool_samples", "mse", "r2", "r2_retention"]].copy()
    lines = []
    for _, row in compact.iterrows():
        lines.append(
            f"{row['model']} & {row['run_id']} & {int(row['train_gms_used']) if not pd.isna(row['train_gms_used']) else '-'} "
            f"& {int(row['use_scales_count']) if not pd.isna(row['use_scales_count']) else '-'} "
            f"& {format_samples(row['pool_samples'])} & {format_float(row['mse'], 4)} "
            f"& {format_float(row['r2'], 4)} & {format_float(row['r2_retention'], 4)} \\\\"
        )
    conclusion_text = "\n".join([f"\\item {c}" for c in conclusions])
    latex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\title{{O2 Data Efficiency Report}}
\author{{Seismic-FNO Experiments}}
\date{{{datetime.now().strftime('%Y-%m-%d')}}}
\begin{{document}}
\maketitle

\section*{{Protocol}}
The O2 study keeps the fixed 474-GM test set from O1 unchanged and prunes only
the supervised train/validation pool. FNO-Large, Transformer, and BiLSTM are
trained with the O1-final hyperparameters. E-Base is imported from O1 full-data
results.

\section*{{Key Conclusions}}
\begin{{itemize}}
{conclusion_text}
\end{{itemize}}

\section*{{Figures}}
\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{figures/fig1_mse_degradation.png}}\caption{{Test MSE degradation.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{figures/fig2_r2_retention.png}}\caption{{R2 retention relative to each model's E-Base.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{figures/fig3_retention_heatmap.png}}\caption{{Retention heatmap over GM and AC budgets.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{figures/fig4_training_curves.png}}\caption{{Representative validation curves.}}\end{{figure}}

\section*{{Quantitative Table}}
\small
\begin{{tabular}}{{llrrrrrr}}
\toprule
Model & Run & GMs & ACs & Pool & MSE & $R^2$ & $R^2$ Ret. \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{document}}
"""
    report_path.write_text(latex.strip() + "\n", encoding="utf-8")


def write_pdf(report_dir: Path, conclusions: list[str]) -> None:
    pdf_path = report_dir / "O2_data_efficiency_report.pdf"
    fig_dir = report_dir / "figures"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.06, 0.92, "O2 Data Efficiency Report", fontsize=24, weight="bold")
        fig.text(0.06, 0.87, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10)
        protocol = (
            "Fixed evaluation set: 474 held-out GMs x 57 ACs = 27,018 samples.\n"
            "Reduced supervision varies GM/source diversity and AC/intensity scales.\n"
            "Models: FNO-Large, Transformer, BiLSTM. E-Base imported from O1."
        )
        fig.text(0.06, 0.78, protocol, fontsize=12)
        y = 0.66
        for item in conclusions:
            wrapped = textwrap.fill(item, width=100)
            fig.text(0.08, y, f"- {wrapped}", fontsize=10)
            y -= 0.09
        pdf.savefig(fig)
        plt.close(fig)
        for name in [
            "fig1_mse_degradation.png",
            "fig2_r2_retention.png",
            "fig3_retention_heatmap.png",
            "fig4_training_curves.png",
            "fig5_results_table.png",
        ]:
            path = fig_dir / name
            if not path.exists():
                continue
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate O2 data-efficiency report.")
    parser.add_argument("--results_root", default=str(O2_DIR / "results"))
    parser.add_argument("--compiled_csv", default=None)
    parser.add_argument("--report_dir", default=str(O2_DIR / "results" / "report"))
    args = parser.parse_args()

    results_root = Path(args.results_root)
    compiled_csv = Path(args.compiled_csv) if args.compiled_csv else results_root / "compiled" / "o2_data_efficiency_metrics.csv"
    if not compiled_csv.exists():
        raise SystemExit(f"Compiled metrics not found: {compiled_csv}")

    report_dir = Path(args.report_dir)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = sorted_df(ensure_numeric(pd.read_csv(compiled_csv)))
    df.to_csv(report_dir / "o2_report_summary.csv", index=False)

    plot_mse_by_axis(df, fig_dir)
    plot_retention(df, fig_dir)
    plot_heatmaps(df, fig_dir)
    plot_training_curves(results_root, fig_dir)
    plot_summary_table(df, fig_dir)
    conclusions = build_conclusions(df)
    (report_dir / "o2_report_conclusions.json").write_text(json.dumps(conclusions, indent=2), encoding="utf-8")
    write_markdown(df, report_dir, conclusions)
    write_latex(df, report_dir, conclusions)
    write_pdf(report_dir, conclusions)
    print(f"Wrote O2 report to {report_dir}")


if __name__ == "__main__":
    main()
