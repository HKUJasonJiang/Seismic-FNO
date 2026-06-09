"""Analyze O3 ground-motion OOD descriptors using Fourier and response spectra."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def load_gm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    return data[:, 0], data[:, 1]


def arias_duration(time: np.ndarray, acc: np.ndarray) -> tuple[float, float]:
    if len(time) < 2:
        return 0.0, 0.0
    dt = float(np.median(np.diff(time)))
    energy = np.cumsum(acc**2) * dt
    total = float(energy[-1])
    if total <= 0:
        return 0.0, 0.0
    norm = energy / total
    t05 = float(time[np.searchsorted(norm, 0.05)])
    t95 = float(time[min(np.searchsorted(norm, 0.95), len(time) - 1)])
    return total, max(t95 - t05, 0.0)


def fourier_features(time: np.ndarray, acc: np.ndarray) -> dict[str, float]:
    if len(time) < 2:
        return {}
    dt = float(np.median(np.diff(time)))
    freqs = np.fft.rfftfreq(len(acc), d=dt)
    amp = np.abs(np.fft.rfft(acc))
    amp[0] = 0.0
    total = float(np.sum(amp) + 1e-12)
    power = amp**2
    power_total = float(np.sum(power) + 1e-12)
    dominant_freq = float(freqs[int(np.argmax(amp))])
    centroid = float(np.sum(freqs * amp) / total)

    def band_power(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(power[mask]) / power_total)

    return {
        "dominant_freq_hz": dominant_freq,
        "spectral_centroid_hz": centroid,
        "band_0p1_1_hz": band_power(0.1, 1.0),
        "band_1_5_hz": band_power(1.0, 5.0),
        "band_5_10_hz": band_power(5.0, 10.0),
        "band_10_25_hz": band_power(10.0, 25.0),
    }


def response_features(psa: pd.DataFrame, sa: pd.DataFrame, name: str) -> dict[str, float]:
    periods = psa["Period (s)"].to_numpy()
    p = psa[name].to_numpy()
    s = sa[name].to_numpy()

    def interp(period: float, values: np.ndarray) -> float:
        return float(np.interp(period, periods, values))

    def mean_range(lo: float, hi: float, values: np.ndarray) -> float:
        mask = (periods >= lo) & (periods <= hi)
        return float(np.mean(values[mask])) if np.any(mask) else float("nan")

    feats = {
        "sa_t0_ms2": float(s[0]),
        "psa_t0p2_ms2": interp(0.2, p),
        "psa_t0p5_ms2": interp(0.5, p),
        "psa_t1p0_ms2": interp(1.0, p),
        "psa_t2p0_ms2": interp(2.0, p),
        "psa_mean_0p1_0p5": mean_range(0.1, 0.5, p),
        "psa_mean_0p5_1p5": mean_range(0.5, 1.5, p),
        "psa_mean_1p5_4p0": mean_range(1.5, 4.0, p),
    }
    norm = max(abs(feats["sa_t0_ms2"]), 1e-12)
    for key in list(feats):
        if key.startswith("psa_"):
            feats[f"{key}_norm"] = feats[key] / norm
    return feats


def percentile_rank(values: np.ndarray, query: float) -> float:
    return float(100.0 * np.mean(values <= query))


def descriptor_columns(normalize_amplitude: bool) -> list[str]:
    if normalize_amplitude:
        return [
            "rms_acc_norm",
            "arias_proxy_norm",
            "duration_5_95_s",
            "dominant_freq_hz",
            "spectral_centroid_hz",
            "band_0p1_1_hz",
            "band_1_5_hz",
            "band_5_10_hz",
            "band_10_25_hz",
            "psa_t0p2_ms2_norm",
            "psa_t0p5_ms2_norm",
            "psa_t1p0_ms2_norm",
            "psa_t2p0_ms2_norm",
            "psa_mean_0p1_0p5_norm",
            "psa_mean_0p5_1p5_norm",
            "psa_mean_1p5_4p0_norm",
        ]
    return [
        "pga_ms2",
        "rms_acc_ms2",
        "arias_proxy",
        "duration_5_95_s",
        "dominant_freq_hz",
        "spectral_centroid_hz",
        "band_0p1_1_hz",
        "band_1_5_hz",
        "band_5_10_hz",
        "band_10_25_hz",
        "sa_t0_ms2",
        "psa_t0p2_ms2",
        "psa_t0p5_ms2",
        "psa_t1p0_ms2",
        "psa_t2p0_ms2",
        "psa_mean_0p1_0p5",
        "psa_mean_0p5_1p5",
        "psa_mean_1p5_4p0",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze O3 GM OOD descriptors.")
    parser.add_argument("--input_root", default="6-code/O3-generalization/results/gm_ood/inputs_scale09")
    parser.add_argument("--spectra_dir", default="6-code/O3-generalization/results/gm_ood/response_spectra_scale09")
    parser.add_argument("--output_dir", default="6-code/O3-generalization/results/gm_ood/analysis_scale09")
    parser.add_argument("--normalize_amplitude", action="store_true", help="Use PGA-normalized waveform/spectrum-shape descriptors for OOD distance.")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    gm_dir = input_root / "GMs"
    manifest = pd.read_csv(input_root / "gm_manifest.csv")
    psa = pd.read_csv(Path(args.spectra_dir) / "pSa.csv")
    sa = pd.read_csv(Path(args.spectra_dir) / "Sa.csv")
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in manifest.iterrows():
        name = str(row["name"])
        filename = str(row["filename"])
        time, acc = load_gm(gm_dir / filename)
        pga = float(np.max(np.abs(acc)))
        acc_norm = acc / max(pga, 1e-12)
        arias, duration = arias_duration(time, acc)
        arias_norm, _ = arias_duration(time, acc_norm)
        base = {
            "name": name,
            "filename": filename,
            "group": row["group"],
            "source": row["source"],
            "gm_id": row["gm_id"],
            "scale_id": row["scale_id"],
            "descriptor_mode": "amplitude_normalized" if args.normalize_amplitude else "absolute",
            "pga_ms2": pga,
            "rms_acc_ms2": float(np.sqrt(np.mean(acc**2))),
            "arias_proxy": arias,
            "rms_acc_norm": float(np.sqrt(np.mean(acc_norm**2))),
            "arias_proxy_norm": arias_norm,
            "duration_5_95_s": duration,
            "n_steps": len(acc),
            "dt": float(np.median(np.diff(time))) if len(time) > 1 else float("nan"),
        }
        base.update(fourier_features(time, acc_norm if args.normalize_amplitude else acc))
        base.update(response_features(psa, sa, name))
        rows.append(base)

    desc = pd.DataFrame(rows)
    descriptor_cols = descriptor_columns(args.normalize_amplitude)
    train = desc[desc["group"] == "train_pool"].copy()
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[descriptor_cols])
    all_x = scaler.transform(desc[descriptor_cols])

    pca = PCA(n_components=2, random_state=42)
    pca.fit(train_x)
    pca_all = pca.transform(all_x)
    desc["pc1"] = pca_all[:, 0]
    desc["pc2"] = pca_all[:, 1]

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(train_x)
    train_dist = nn.kneighbors(train_x, return_distance=True)[0][:, 1]
    all_dist = nn.kneighbors(all_x, n_neighbors=1, return_distance=True)[0][:, 0]
    desc["train_nn_distance"] = all_dist
    desc["train_nn_percentile"] = [percentile_rank(train_dist, v) for v in all_dist]

    output_dir.mkdir(parents=True, exist_ok=True)
    desc.to_csv(output_dir / "gm_descriptors.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    classical = desc[desc["group"] == "classical"].copy()
    classical.to_csv(output_dir / "classical_gm_ood_scores.csv", index=False)

    title_suffix = " (PGA-normalized)" if args.normalize_amplitude else ""
    colors = {"train_pool": "#8c8c8c", "fixed_test": "#1f77b4", "classical": "#d62728"}
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    for group, gdf in desc.groupby("group"):
        ax.scatter(gdf["pc1"], gdf["pc2"], s=12 if group != "classical" else 70,
                   alpha=0.35 if group != "classical" else 0.95,
                   label=group, color=colors.get(group, "black"),
                   edgecolor="black" if group == "classical" else "none")
    for _, r in classical.iterrows():
        ax.text(r["pc1"], r["pc2"], str(r["name"]).split("-")[0], fontsize=8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("GM OOD Descriptor PCA" + title_suffix)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "gm_ood_pca.png", dpi=220)
    fig.savefig(fig_dir / "gm_ood_pca.pdf")
    plt.close(fig)

    periods = psa["Period (s)"].to_numpy()
    train_names = desc.loc[desc["group"] == "train_pool", "name"].tolist()
    test_names = desc.loc[desc["group"] == "fixed_test", "name"].tolist()
    classical_names = classical["name"].tolist()
    train_mat = psa[train_names].to_numpy()
    test_mat = psa[test_names].to_numpy()
    if args.normalize_amplitude:
        train_den = sa.loc[sa["Period (s)"] == 0, train_names].iloc[0].to_numpy()
        test_den = sa.loc[sa["Period (s)"] == 0, test_names].iloc[0].to_numpy()
        train_mat = train_mat / np.maximum(np.abs(train_den), 1e-12)
        test_mat = test_mat / np.maximum(np.abs(test_den), 1e-12)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(periods, np.median(train_mat, axis=1), color="#555555", linewidth=2, label="Train median")
    ax.fill_between(periods, np.percentile(train_mat, 5, axis=1), np.percentile(train_mat, 95, axis=1),
                    color="#aaaaaa", alpha=0.25, label="Train 5-95%")
    ax.plot(periods, np.median(test_mat, axis=1), color="#1f77b4", linewidth=1.8, label="Test median")
    for name in classical_names:
        values = psa[name].to_numpy()
        if args.normalize_amplitude:
            den = float(sa.loc[sa["Period (s)"] == 0, name].iloc[0])
            values = values / max(abs(den), 1e-12)
        ax.plot(periods, values, linewidth=2, label=name.split("-")[0])
    ax.set_xlim(0.05, 6.0)
    ax.set_yscale("log")
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("PSa / PGA" if args.normalize_amplitude else "PSa (m/s2)")
    ax.set_title("Nigam-Jennings Pseudo-Acceleration Spectra" + title_suffix)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "psa_overlay.png", dpi=220)
    fig.savefig(fig_dir / "psa_overlay.pdf")
    plt.close(fig)

    train = desc[desc["group"] == "train_pool"].copy()
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.5))
    metrics = (
        ["rms_acc_norm", "spectral_centroid_hz", "psa_t1p0_ms2_norm", "train_nn_distance"]
        if args.normalize_amplitude
        else ["pga_ms2", "spectral_centroid_hz", "duration_5_95_s", "train_nn_distance"]
    )
    for ax, metric in zip(axs.ravel(), metrics):
        ax.hist(train[metric], bins=40, color="#999999", alpha=0.65, label="Train")
        ax.hist(desc.loc[desc["group"] == "fixed_test", metric], bins=40, color="#1f77b4", alpha=0.35, label="Test")
        for _, r in classical.iterrows():
            ax.axvline(r[metric], color="#d62728", linewidth=1.4)
        ax.set_title(metric)
        ax.grid(True, alpha=0.2)
    axs[0, 0].legend(frameon=False)
    fig.suptitle("OOD Descriptor Marginals" + title_suffix)
    fig.tight_layout()
    fig.savefig(fig_dir / "descriptor_histograms.png", dpi=220)
    fig.savefig(fig_dir / "descriptor_histograms.pdf")
    plt.close(fig)

    summary_cols = ["name", "pga_ms2", "spectral_centroid_hz"]
    summary_cols.append("psa_t1p0_ms2_norm" if args.normalize_amplitude else "psa_t1p0_ms2")
    summary_cols.extend(["train_nn_distance", "train_nn_percentile"])
    summary = classical[summary_cols]
    print(summary.to_string(index=False))
    print(f"Wrote O3 GM OOD analysis to {output_dir}")


if __name__ == "__main__":
    main()
