# O3 Generalization Status and TODO

Last organized: 2026-06-08

## One-Line Status

O3-A GM descriptor OOD, O3-B classical-GM zero-shot inference, and O3-C PGA
extrapolation OOD are complete. O3-D sampling-resolution OOD is designed but not
implemented or run. A future FNO spectral-bandwidth ablation is recorded as a
follow-up idea, not part of the completed O3 results yet.

## Directory Map

Top-level O3 directory:

- `README.md`: short entry point and document index.
- `docs/00_status_todo.md`: this master status/TODO file.
- `docs/01_overall_experiment_plan.md`: full O3 plan across all OOD axes.
- `docs/02_sampling_resolution_todo.md`: detailed sampling-resolution OOD TODO.
- `docs/99_code_modify_log.md`: code change log placeholder.
- `scripts/`: runnable Python scripts.
- `tools/CalResponseSpectra_src/`: C++ response-spectrum source and Linux binaries.
- `results/`: generated outputs; do not manually edit.

## O3-A: GM Descriptor OOD

Purpose:

- Compare 4 classical GMs against the 3,000-GM train pool and 474-GM fixed test
  set using waveform and response-spectrum descriptors.
- Use normalized descriptors to separate waveform/spectral shape OOD from
  absolute-amplitude/PGA OOD.

Status:

- [x] Prepared K-NET and classical GM text inputs.
- [x] Compiled Linux C++ response-spectrum executable.
- [x] Ran Nigam-Jennings response spectra for 3,474 K-NET GMs + 4 classical GMs.
- [x] Generated absolute-amplitude OOD analysis.
- [x] Generated amplitude-normalized OOD analysis.
- [x] Generated PCA, PSa overlay, and descriptor histogram figures.

Key normalized OOD result:

| Classical GM | Shape-OOD percentile vs train pool |
|---|---:|
| El Centro | 34.8% |
| San Fernando | 39.8% |
| Michoacan | 82.7% |
| Kobe | 62.8% |

Main outputs:

- `results/gm_ood/analysis_scale09_normalized/classical_gm_ood_scores.csv`
- `results/gm_ood/analysis_scale09_normalized/gm_descriptors.csv`
- `results/gm_ood/analysis_scale09_normalized/figures/gm_ood_pca.png`
- `results/gm_ood/analysis_scale09_normalized/figures/psa_overlay.png`
- `results/gm_ood/analysis_scale09_normalized/figures/descriptor_histograms.png`

Notes:

- Classical GM descriptor analysis used native sample intervals where available:
  50Hz, 100Hz, and 200Hz records were not forced to 50Hz for response-spectrum
  descriptor computation.

## O3-B: Classical-GM Zero-Shot Inference

Purpose:

- Evaluate the O1 full-data checkpoints on the 4 classical GMs.

Status:

- [x] Ran BiLSTM, Transformer, and FNO-Large on 4 classical GMs.
- [x] Generated time-history overlays.
- [x] Generated event-level metric bar charts.
- [x] Ran native-GM-input inference for 50/100/200Hz classical records.
- [x] Generated native-GM-input metric table and figures.

Important caveat:

- [x] Recorded caveat: the original comparison was normalized to 50Hz / 3000 points.
- [x] Native GM input inference is done.
- [ ] True native-resolution response-GT inference is not done yet because the
  current roof response files are all 50Hz.

Main outputs:

- `results/classical_gms_o1_full/classical_gm_metrics.csv`
- `results/classical_gms_o1_full/classical_time_histories.png`
- `results/classical_gms_o1_full/classical_metric_bars.png`
- `results/classical_gms_native_input_o1_full/classical_native_input_metrics.csv`
- `results/classical_gms_native_input_o1_full/classical_native_input_time_histories.png`
- `results/classical_gms_native_input_o1_full/classical_native_input_metric_bars.png`

Result summary:

- BiLSTM is strongest overall on the normalized-50Hz classical GM inference.
- Transformer is generally second.
- FNO-Large does not beat Transformer/BiLSTM on this completed comparison.
- On native-GM-input inference, FNO-Large is strongest for the higher-resolution
  Kobe 100Hz and Michoacan 200Hz records when predictions are interpolated to
  the available 50Hz response-GT time axis.

## O3-C: PGA Extrapolation OOD

Purpose:

- Train each model only on the 1-10 m/s2 PGA scale range.
- Evaluate each trained model on the unchanged 474-GM x 57-scale fixed test
  grid.

Status:

- [x] Implemented `scripts/train_eval_pga_ood.py`.
- [x] Trained FNO-Large `PGA-1to10`.
- [x] Trained Transformer `PGA-1to10`.
- [x] Trained BiLSTM `PGA-1to10`.
- [x] Evaluated full 474 x 57 test grid.
- [x] Evaluated metrics by AC/PGA scale.
- [x] Generated review figures: training curves, PGA degradation curves,
  accuracy-cost bars, and test example overlays.

Overall fixed-test results:

| Model | MSE | R2 | Train time [s] | Peak GPU [GB] |
|---|---:|---:|---:|---:|
| BiLSTM | 0.01476 | 0.9810 | 17620 | 15.61 |
| Transformer | 0.03190 | 0.9590 | 7357 | 2.54 |
| FNO-Large | 0.06694 | 0.9140 | 7271 | 3.30 |

Main outputs:

- `results/pga_ood/runs/{model}/PGA-1to10/training_log.csv`
- `results/pga_ood/runs/{model}/PGA-1to10/evaluation_overall.csv`
- `results/pga_ood/runs/{model}/PGA-1to10/evaluation_by_scale.csv`
- `results/figures_o3_review/pga_ood_training_curves.png`
- `results/figures_o3_review/pga_ood_by_scale_mse_r2.png`
- `results/figures_o3_review/pga_ood_overall_cost_summary.png`
- `results/figures_o3_review/pga_ood_test_examples_time_histories.png`
- `results/figures_o3_review/pga_ood_test_examples_zoom.png`

Result summary:

- FNO-Large is not the best in PGA extrapolation OOD.
- Transformer is stronger than FNO-Large.
- BiLSTM is strongest but much more expensive in GPU memory and training time.

## O3-D: Sampling-Resolution OOD

Purpose:

- Test whether FNO-Large shows an operator-learning advantage when training
  sampling resolution differs from evaluation resolution.

Current status:

- [x] Designed experiment matrix.
- [x] Wrote detailed TODO and caveats in `docs/02_sampling_resolution_todo.md`.
- [x] Implemented sampling-aware dataset/resampling wrapper.
- [x] Implemented `scripts/train_eval_sampling_ood.py`.
- [x] Implemented `scripts/run_sampling_ood_queue.py`.
- [ ] Implement `scripts/evaluate_classical_sampling_ood.py`.
- [x] Implemented `scripts/plot_sampling_ood.py`.
- [x] Smoke-tested sampling wrapper and model forward/backward.
- [x] Launched main 9-run matrix.
- [x] Generate sampling-OOD figures.
- [ ] Generate sampling-OOD report.

Running queues:

| Queue | GPU | PID | Log | Jobs |
|---|---:|---:|---|---|
| BiLSTM sampling | 0 | 2737012 | `results/logs/o3_sampling_gpu0_bilstm.log` | `S-20Hz -> S-25Hz -> S-50Hz` |
| FNO-Large sampling | 1 | 2737013 | `results/logs/o3_sampling_gpu1_fno.log` | `S-20Hz -> S-25Hz -> S-50Hz` |
| Transformer sampling | 1 | 2737014 | `results/logs/o3_sampling_gpu1_transformer.log` | `S-20Hz -> S-25Hz -> S-50Hz` |

First observed status:

- FNO-Large `S-20Hz` epoch 1 completed: val MSE `0.06988`, val R2 `0.9110`.
- Transformer `S-20Hz` epoch 1 completed: val MSE `0.17647`, val R2 `0.7753`.
- BiLSTM `S-20Hz` was still inside epoch 1 at launch check.

Main run matrix:

| Run ID | Train Hz | Train length | rFFT bins | Fixed test |
|---|---:|---:|---:|---|
| `S-20Hz` | 20 | 1200 | 601 | 50Hz / 3000 |
| `S-25Hz` | 25 | 1500 | 751 | 50Hz / 3000 |
| `S-50Hz` | 50 | 3000 | 1501 | 50Hz / 3000 |

Optional stress run:

| Run ID | Train Hz | Train length | rFFT bins | Fixed test |
|---|---:|---:|---:|---|
| `S-10Hz` | 10 | 600 | 301 | 50Hz / 3000 |

Important design note:

- `S-10Hz` is optional because it removes original 5-25Hz information before
  training. It measures extreme coarse-to-fine or super-resolution behavior, not
  ordinary sampling OOD.

## Future Follow-Up: FNO Spectral-Bandwidth Ablation

Purpose:

- Test whether FNO underperformance is caused by the `n_modes=512` Fourier
  bandwidth limit.

Status:

- [x] Hypothesis recorded.
- [ ] Not implemented.
- [ ] Not run.

Candidate matrix:

- FNO-Large `n_modes=512`
- FNO-Large `n_modes=1024`
- FNO-Large near-full 50Hz spectrum, e.g. `n_modes=1500`

Reasoning:

- For 60s records, `n_modes=512` covers about `8.5Hz`.
- At 50Hz sampling, Nyquist is `25Hz`, so high-frequency peak information may
  be truncated.
- This may explain why FNO-Large loses to Transformer/BiLSTM on fixed 50Hz
  O1/O3 metrics.

## Cleanup TODO

- [ ] Remove or archive stale PID files under `results/pids/` after confirming
  no related processes are running.
- [ ] Decide whether to keep `results/gm_ood/analysis_scale09/` absolute-amplitude
  analysis or mark it as superseded by normalized analysis in the report.
- [ ] Remove temporary smoke-test spectra under `results/gm_ood/smoke/` if no
  longer needed.
- [ ] Move final paper-ready figures into a single report figure folder after
  O3-D is complete.

## Next Recommended Work Order

1. Implement O3-D sampling-resolution dataset wrapper and smoke test.
2. Run main O3-D matrix: `S-20Hz`, `S-25Hz`, `S-50Hz` for all three models.
3. Generate O3-D report and compare with O3-B/C.
4. If FNO still underperforms, run FNO spectral-bandwidth ablation.
5. Write final O3 generalization report using the completed A/B/C/D blocks.

