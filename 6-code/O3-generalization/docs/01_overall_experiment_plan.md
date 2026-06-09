# O3 Generalization Experiment Plan

## Objective

Evaluate whether the three O1/O2 finalist models generalize outside the fixed
in-distribution 474-GM split:

- FNO-Large
- Transformer
- BiLSTM

O3 has three complementary OOD axes:

- Ground-motion waveform OOD: four classical GMs versus K-NET train/test GMs.
- PGA extrapolation OOD: train only on lower-PGA AC combinations, then evaluate
  on the full 474 x 57 fixed test grid.
- Sampling-resolution OOD: train at 10/25/50Hz and evaluate at the fixed 50Hz
  474-GM test grid plus four classical GMs.

## Part A: GM OOD Descriptor Analysis

Before evaluating predictions on the four classical GMs, quantify whether those
records are outside the K-NET distribution.

Reference groups:

- K-NET train pool: 3,000 GMs from the canonical seed-42 split.
- K-NET fixed test set: 474 held-out GMs from the same split.
- Classical GMs: the four records supplied under `6-code/data/classical_GM`
  once available.

Descriptor families:

- Time-domain: PGA, Arias intensity, RMS acceleration, duration proxies.
- Fourier spectrum: log-amplitude spectrum, low/mid/high frequency energy
  bands, dominant frequency, spectral centroid.
- Response spectrum: elastic response spectrum from the user-provided exe.

Required outputs:

- `gm_descriptors.csv`: one row per unscaled GM.
- `classical_gm_ood_scores.csv`: nearest-neighbor distance and percentile
  against train/test distributions.
- PCA/UMAP scatter: train, test, classical records.
- Distribution plots: PGA, dominant frequency, spectral centroid, response
  spectrum ordinates.
- Mean response spectrum overlays with classical records highlighted.

Interpretation target:

- Decide whether each classical GM is in-distribution, marginal OOD, or strong
  OOD relative to the 3,000-GM training pool and 474-GM fixed test set.

## Part B: Zero-Shot Classical-GM Evaluation

Use the full-data O1 checkpoints unless O2/O3-specific checkpoints are requested.

Models:

- FNO-Large: `O1-baseline-comparison/results/runs/FNO-Large-bs64/...`
- Transformer: `O1-baseline-comparison/results/runs/Transformer/...`
- BiLSTM: `O1-baseline-comparison/results/runs/BiLSTM/...`

Evaluation records:

- 1940 El Centro
- 1971 San Fernando
- 1985 Michoacan
- 1991 Kobe

Metrics:

- Time-history MSE/RMSE/MAE/R2.
- PFA error and PFA R2 where available.
- Fourier spectral R2/log-MSE.
- Response-spectrum error once the response-spectrum exe is available.

Figures:

- Time-history overlays per event and model.
- Fourier spectrum overlays.
- Response spectrum overlays.
- Bar chart: event-level error by model.
- OOD score versus prediction error scatter.

## Part C: PGA Extrapolation OOD

Purpose:

- Train each model once using only lower-PGA AC levels.
- Evaluate on the unchanged fixed 474-GM x 57-AC test grid.
- Report degradation by test AC/PGA bin.

Scale-index finding:

- The HDF5 stores only scaled waveforms in `Acc_GMs`.
- Global index is `gm_id * 57 + scale_id`.
- A quick PGA back-calculation shows scale index 46 is approximately
  10 m/s^2. The final script should compute and save this mapping directly from
  the HDF5 rather than hard-coding it.

Recommended O3-PGA training set:

- Train GMs: canonical 3,000-GM pool, split 80/20 train/val.
- Train ACs: scale indices corresponding to 1-10 m/s^2, approximately `9..46`.
- Alternative sensitivity row: `0..46`, i.e. 0.1-10 m/s^2, if we want a broader
  low-to-mid PGA training domain.
- Test ACs: all 57 ACs on the fixed 474-GM test set.

Models:

- FNO-Large
- Transformer
- BiLSTM

Each model is trained once for the selected lower-PGA AC set.

Required outputs:

- `scale_pga_mapping.csv`: scale index, inferred PGA target, train/test flag.
- `pga_ood_training_log.csv` per model.
- `pga_ood_evaluation_by_scale.csv`: metrics grouped by AC scale/PGA.
- `pga_ood_evaluation_overall.csv`: overall fixed-test metrics.

Figures:

- Test MSE/R2 versus PGA/scale index.
- In-domain versus extrapolation shaded region.
- PFA RMSE versus PGA.
- Per-model degradation curves normalized by O1 full-data metrics.


## Part D: Sampling-Resolution OOD

Detailed TODO list:

- `experiment.md`

Purpose:

- Test whether FNO-Large shows an operator-learning advantage when the training
  temporal resolution differs from the evaluation resolution.

Training matrix:

- Train at 10Hz, 25Hz, and 50Hz.
- Use the canonical 3,000-GM pool with the same 80/20 train/validation split.
- Evaluate every run at the fixed 50Hz 474-GM x 57-scale test grid.
- Also evaluate the four classical GMs at 50Hz, with a native-resolution
  classical extension for Kobe/Michoacan where downstream ground truth is
  available.

Models:

- FNO-Large
- Transformer
- BiLSTM

Required outputs:

- Sampling-aware training logs and checkpoints.
- Fixed-test overall and by-scale metrics.
- Classical-GM 50Hz and native-resolution metrics.
- Loss curves, sampling degradation curves, PGA-scale curves, and example
  time-history overlays.

## Compute Plan

O2 is currently occupying both GPUs. O3 should be prepared now but only launched
after O2 queues finish, unless the user explicitly wants to share GPUs.

Recommended launch order after O2:

1. Run GM descriptor analysis on CPU.
2. Run classical-GM zero-shot inference on one GPU; this is short.
3. Train O3-PGA extrapolation models:
   - GPU 0: BiLSTM.
   - GPU 1: FNO-Large then Transformer, or FNO-Large and Transformer in
     separate low-contention queues if benchmarked.
4. Generate O3 report with descriptor/OOD plots and model metrics.

## Report Deliverables

- `O3_generalization_report.md`
- `O3_generalization_report.tex`
- `O3_generalization_report.pdf`
- descriptor CSVs and model metric CSVs
- paper-ready figures for GM OOD and PGA extrapolation

The paper narrative should explicitly separate waveform OOD from PGA
extrapolation OOD. If FNO does better on either axis, this becomes the strongest
argument for operator-learning generalization beyond the in-distribution O1
fixed test split.
