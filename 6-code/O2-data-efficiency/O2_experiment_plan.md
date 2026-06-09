# O2 Data Efficiency Experiment Plan

## Objective

Compare data efficiency for the three models that advance from O1:

- FNO-Large
- Transformer
- BiLSTM

The fixed O1 test protocol remains unchanged: 474 held-out GMs with all 57 amplitude coefficients, giving 27,018 test samples. Only the supervised train/validation pool is reduced.

## Data-Pruning Axes

Use the same sparse-supervision protocol as the manuscript draft:

- Source diversity pruning: reduce the number of training-pool GMs.
- Intensity-scale pruning: reduce the number of amplitude coefficients.
- Joint pruning: reduce both GMs and amplitude coefficients.

The shared split utility `make_fixed_474_split(train_gms_used=..., use_scales_count=...)` already implements this setup:

- shuffle 3,474 GMs with seed 42
- reserve the last 474 GMs as the fixed test set
- select the first `train_gms_used` GMs from the remaining 3,000-GM pool
- split selected GMs into 80 percent train and 20 percent validation
- select ACs by stratified linspace over the 57 scale indices

## Experiment Matrix

| Run ID | Pool GMs | AC Scales | Pool Samples | Purpose |
|---|---:|---:|---:|---|
| E-Base | 3000 | 57 | 171.0k | full-data reference |
| E-AC80 | 3000 | 46 | 138.0k | AC-only pruning |
| E-AC60 | 3000 | 34 | 102.0k | AC-only pruning |
| E-AC40 | 3000 | 23 | 69.0k | AC-only pruning |
| E-AC20 | 3000 | 11 | 33.0k | AC-only pruning |
| E-AC10 | 3000 | 6 | 18.0k | extreme AC-only pruning |
| E-AC05 | 3000 | 3 | 9.0k | extreme AC-only pruning |
| E-GM80 | 2400 | 57 | 136.8k | GM-only pruning |
| E-GM60 | 1800 | 57 | 102.6k | GM-only pruning |
| E-GM40 | 1200 | 57 | 68.4k | GM-only pruning |
| E-GM20 | 600 | 57 | 34.2k | GM-only pruning |
| E-J80 | 2400 | 46 | 110.4k | joint pruning |
| E-J60 | 1800 | 34 | 61.2k | joint pruning |
| E-J40 | 1200 | 23 | 27.6k | joint pruning |
| E-J20 | 600 | 11 | 6.6k | joint pruning |

Each run is repeated for FNO-Large, Transformer, and BiLSTM, yielding 45 total training runs.

## Model Settings

Reuse the O1-final model settings so O2 tests data efficiency rather than architecture retuning.

| Model | Architecture | LR | Batch | Epochs | Scheduler | Notes |
|---|---|---:|---:|---:|---|---|
| FNO-Large | n_modes=512, hidden=64, layers=8, domain_padding=0.1 | 1e-3 | 64 | 50 | StepLR 20, gamma 0.5 | new O2 reference model |
| Transformer | patch=15, d_model=256, heads=8, layers=10, pre-norm | 3e-4 | 64 | 50 | StepLR 20, gamma 0.5 | stabilized O1 rerun setting |
| BiLSTM | O1 BiLSTM baseline | 3e-4 | 64 | 50 | StepLR 20, gamma 0.5 | keep grad_clip=1.0 and NaN guard |

## GPU Schedule

Use two GPUs, with no concurrent models on the same GPU.

- GPU 0: BiLSTM only.
- GPU 1: FNO-Large and Transformer sequentially.

Recommended order:

1. Start BiLSTM queue on GPU 0 for all 15 O2 configurations.
2. Start FNO-Large queue on GPU 1 for all 15 configurations.
3. After FNO-Large finishes, start Transformer queue on GPU 1 for all 15 configurations.

This keeps the expensive BiLSTM isolated while avoiding memory contention for FNO/Transformer.

## Required Records

For every model and O2 run ID, save:

- `config.json`: model, run_id, seed, pool_gms, ac_scales, split indices, lr, batch size, epochs, scheduler, grad_clip.
- `training_log.csv`: epoch metrics, lr, epoch time, peak GPU memory, max grad norm where available.
- `summary.json`: best epoch, best validation MSE/R2/PFA RMSE, train time, peak GPU, checkpoint path, parameter count.
- `evaluation.json/csv`: fixed 474-GM test MSE, RMSE, MAE, R2, PFA metrics, spectral metrics, latency.
- `dataset_indices.pkl` and `split_info.txt`: exact train/val/test indices.

## Main Tables and Figures

Primary table columns:

- Model
- Run ID
- Num GMs
- Num Scales
- Pool Samples
- Test MSE
- Test R2
- PFA RMSE
- Train Time
- Peak GPU GB
- Best Epoch
- Relative R2 retention versus that model's E-Base
- Relative training time versus that model's E-Base

Primary figures:

- Test R2 versus pool samples, one curve per model.
- Test MSE versus pool samples, log-scaled x-axis.
- AC-only sensitivity, GM-only sensitivity, and joint-pruning sensitivity panels.
- Pareto plot: test MSE/R2 versus training time, bubble size = GPU memory.
- Per-model retention heatmap over GM percent and AC percent.

## Expected Interpretation

The key scientific question is not whether every reduced run beats O1, but which model degrades slowest under sparse supervision:

- FNO should be tested for operator-learning sample efficiency, especially under AC sparsification.
- Transformer is now a strong accuracy competitor and may also be data-efficient.
- BiLSTM should be included because its O1 MSE is very low, but its resource cost should be reported transparently.

The paper claim should be model-relative: compare each reduced setting against that model's own full-data E-Base, then compare the retention curves across models.
