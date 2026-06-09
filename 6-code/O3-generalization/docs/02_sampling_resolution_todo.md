# O3 Sampling-Resolution OOD Experiment TODO

## Goal

Test whether the finalist models learn a resolution-robust seismic response
mapping rather than only a fixed `3000 -> 3000` sequence mapping.

This O3 extension uses the same canonical 3,000-GM training pool and 474-GM
held-out test set, but changes the training sampling rate. All fixed-test
evaluation is performed at the native K-NET benchmark resolution:

- Test grid: `50 Hz`, `60 s`, `3000` time steps.
- Test set: fixed `474 GMs x 57 AC scales`.
- Train/validation split: canonical 3,000 GM pool, 80/20 by GM, seed `42`.

## Models

- [ ] FNO-Large
- [ ] Transformer
- [ ] BiLSTM

## Sampling-OOD Matrix

Main runs:

| Run ID | Train sampling | Train length | rFFT bins | Val sampling | Fixed test sampling | Purpose |
|---|---:|---:|---:|---:|---:|---|
| `S-20Hz` | 20 Hz | 1200 | 601 | 20 Hz | 50 Hz / 3000 | Coarse-to-native OOD with most low/mid frequencies preserved |
| `S-25Hz` | 25 Hz | 1500 | 751 | 25 Hz | 50 Hz / 3000 | Moderate coarse-to-native OOD |
| `S-50Hz` | 50 Hz | 3000 | 1501 | 50 Hz | 50 Hz / 3000 | Native-resolution control |

Optional stress run:

| Run ID | Train sampling | Train length | rFFT bins | Val sampling | Fixed test sampling | Purpose |
|---|---:|---:|---:|---:|---:|---|
| `S-10Hz` | 10 Hz | 600 | 301 | 10 Hz | 50 Hz / 3000 | Extreme coarse-to-native / super-resolution stress test |

Total main training jobs:

- [ ] `3 models x 3 main sampling rates = 9 runs`

Optional additional jobs:

- [ ] `3 models x S-10Hz = 3 runs`

## Data Protocol

- [ ] Use the existing HDF5 dataset:
  - GM input: `/data/home/jason/data/SesimicTransformerData/MDOF/All_GMs/GMs_knet_3474_AF_57.h5`
  - Building response: `/data/home/jason/data/SesimicTransformerData/MDOF/knet-250/Data/fno`
- [ ] Keep the same canonical GM split as O1/O3:
  - 3,000 train/val GMs.
  - 474 fixed test GMs.
  - 57 AC scales.
- [ ] Downsample both input GM and output roof acceleration for training and validation:
  - `S-20Hz`: resample from 50Hz to 20Hz on the same 60s physical time grid.
  - `S-25Hz`: keep every 2nd point from 50Hz.
  - `S-50Hz`: no downsampling.
  - `S-10Hz`: keep every 5th point from 50Hz, optional stress run only.
- [ ] Use an anti-aliasing resampling path for non-integer or lossy downsampling.
  - `25Hz` can be simple decimation by 2.
  - `10Hz` can be decimation by 5 after filtering.
  - `20Hz` should use time-grid interpolation or polyphase resampling, not a naive integer stride.
- [ ] Do not change physical duration: all samples remain 60 seconds.
- [ ] For fixed-test evaluation, input GM is provided at 50Hz and target response is 50Hz.
- [ ] Save exact train/val/test indices and sampling config per run.

## Evaluation Protocol

### Fixed 474-GM Test

For every trained model/checkpoint:

- [ ] Infer on the full fixed test grid: `474 GMs x 57 AC scales` at `50Hz`.
- [ ] Report overall metrics:
  - MSE
  - RMSE
  - MAE
  - R2
  - PFA RMSE / MAE / R2
  - spectral log-MSE
  - high-frequency log-MSE
  - latency per sample
- [ ] Report metrics grouped by AC scale/PGA:
  - MSE vs scale
  - R2 vs scale
  - PFA RMSE vs scale

### Classical GM Inference

Use the four existing classical GMs:

- 1940 El Centro, 50Hz
- 1971 San Fernando, 50Hz
- 1985 Michoacan, 200Hz
- 1991 Kobe, 100Hz

Required comparison:

- [ ] 50Hz normalized classical evaluation:
  - Downsample all classical GM and roof responses to 50Hz.
  - Compare against the same 50Hz output grid as the fixed test.
  - This is directly comparable with the previous O3 classical-GM results.

Resolution-generalization extension:

- [ ] Native-resolution classical evaluation:
  - El Centro/San Fernando at 50Hz.
  - Kobe at 100Hz.
  - Michoacan at 200Hz.
  - Use native-resolution roof response as ground truth.
  - If a model cannot directly produce native-length output, document the
    required interpolation/resampling step and mark it separately from direct
    native inference.

## FNO Spectral-Bandwidth Notes

For a 60s record, frequency resolution is `1/60 = 0.0167Hz`. With
`n_modes=512`, FNO-Large covers roughly `512/60 = 8.5Hz` of physical
frequency content regardless of the input sequence length.

Implications:

- At `50Hz`, Nyquist is `25Hz`, and `512/1501` rFFT bins covers only about
  one third of the available spectrum. High-frequency peak information may be
  truncated.
- At `25Hz`, Nyquist is `12.5Hz`, and `512/751` covers most but not all bins.
- At `20Hz`, Nyquist is `10Hz`, and `512/601` covers almost the full available
  spectrum while still avoiding the degenerate full-spectrum case.
- At `10Hz`, Nyquist is only `5Hz`, and `512 > 301`; the FNO spectral layer is
  effectively full-band for the downsampled signal, but the original `5-25Hz`
  content has already been removed. This run measures extreme super-resolution
  behavior rather than ordinary sampling OOD.

Record this explicitly in configs and reports so that sampling OOD is not
confused with a future FNO `n_modes`/bandwidth ablation.

## Model Fairness Checks

- [ ] FNO-Large must support variable-length inference directly.
- [ ] Transformer positional encoding must be compatible with variable-length
  inference; avoid fixed learned embeddings.
- [ ] BiLSTM naturally supports variable sequence length, but long native
  sequences may be slower; record latency.
- [ ] Use the same optimizer settings family as O3 PGA OOD unless instability
  appears:
  - FNO-Large LR: `1e-3`
  - Transformer LR: `3e-4`
  - BiLSTM LR: `3e-4`
  - AdamW weight decay: `1e-4`
  - grad clip for Transformer/BiLSTM: `1.0`
- [ ] Keep epoch count at `50` for first full run.
- [ ] Save non-finite failure reports if NaN/Inf appears.

## Outputs

Directory:

- [ ] `6-code/O3-generalization/results/sampling_ood/`

Per run:

- [ ] `runs/{model}/{run_id}/config.json`
- [ ] `runs/{model}/{run_id}/training_log.csv`
- [ ] `runs/{model}/{run_id}/summary.json`
- [ ] `runs/{model}/{run_id}/checkpoints/{model}_{run_id}_best.pt`
- [ ] `runs/{model}/{run_id}/evaluation_overall.csv`
- [ ] `runs/{model}/{run_id}/evaluation_by_scale.csv`
- [ ] `runs/{model}/{run_id}/classical_50hz_metrics.csv`
- [ ] `runs/{model}/{run_id}/classical_native_metrics.csv`

Figures:

- [ ] Training/validation MSE curves across `S-20Hz`, `S-25Hz`, `S-50Hz`, with optional `S-10Hz` shown separately as stress test.
- [ ] Fixed-test MSE/R2 vs train sampling rate.
- [ ] Fixed-test MSE/R2 vs PGA scale for each sampling rate.
- [ ] Classical GM time-history overlays at 50Hz.
- [ ] Classical GM native-resolution overlays.
- [ ] Accuracy-cost plot:
  - MSE vs training time.
  - MSE vs peak GPU memory.
  - MSE vs latency.

Report:

- [ ] `O3_sampling_ood_report.md`
- [ ] `O3_sampling_ood_report.tex`
- [ ] `O3_sampling_ood_report.pdf`

## Interpretation Targets

- [ ] Does FNO-Large degrade less than Transformer/BiLSTM when trained at
  `20Hz` or `25Hz` and evaluated at `50Hz`?
- [ ] Does FNO-Large benefit more from increasing train resolution
  `20Hz -> 25Hz -> 50Hz`?
- [ ] Does optional `S-10Hz` fail because high-frequency information is absent,
  or can any model reconstruct useful 50Hz response trends from very coarse input?
- [ ] Does Transformer remain strongest even under coarse-to-native sampling
  transfer?
- [ ] Does BiLSTM remain accurate but expensive under resolution transfer?
- [ ] On Kobe/Michoacan native-resolution tests, does FNO-Large show a clearer
  operator-learning advantage than in fixed-resolution O1/O3?

## Launch TODO

- [ ] Implement a sampling-aware dataset wrapper or transform:
  - Takes original 50Hz HDF5 sample.
  - Downsamples input/target for train/val.
  - Leaves fixed test at 50Hz.
- [ ] Implement `train_eval_sampling_ood.py`.
- [ ] Implement `evaluate_classical_sampling_ood.py`.
- [ ] Implement `plot_sampling_ood.py`.
- [ ] Smoke test one tiny run:
  - model: FNO-Large
  - run: `S-25Hz`
  - epochs: 1
  - small subset
- [ ] Launch full 9-run main matrix.
- [ ] Launch optional `S-10Hz` stress runs only after main matrix is understood.
- [ ] Generate figures and report.

