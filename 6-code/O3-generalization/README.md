# O3 Generalization

This folder contains the O3 generalization experiments for the Seismic-FNO project.
Use the status file first; it is the single source of truth for what has been done
and what remains.

## Start Here

- `docs/00_status_todo.md`: master status, completed results, and next TODOs.
- `docs/01_overall_experiment_plan.md`: full O3 experimental plan.
- `docs/02_sampling_resolution_todo.md`: detailed sampling-resolution OOD design.
- `docs/99_code_modify_log.md`: code modification log placeholder.

## Completed Blocks

- O3-A GM descriptor OOD: completed.
- O3-B classical-GM zero-shot inference at normalized 50Hz: completed.
- O3-C PGA extrapolation OOD: completed.

## Pending Blocks

- O3-D sampling-resolution OOD: designed, not implemented or run.
- FNO spectral-bandwidth ablation: recorded as follow-up, not implemented or run.

## Important Result Folders

- `results/gm_ood/analysis_scale09_normalized/`: normalized GM descriptor OOD.
- `results/classical_gms_o1_full/`: classical GM 50Hz inference results.
- `results/classical_gms_native_input_o1_full/`: native-GM-input classical results, compared to available 50Hz response GT.
- `results/pga_ood/runs/`: PGA OOD model runs.
- `results/figures_o3_review/`: generated O3 PGA/classical review figures.

## Scripts

- `scripts/prepare_o3_gm_inputs.py`: prepare K-NET/classical GM inputs for spectra.
- `scripts/analyze_o3_gm_ood.py`: descriptor and OOD-score analysis.
- `scripts/train_eval_pga_ood.py`: PGA extrapolation OOD training/evaluation.
- `scripts/plot_o3_results.py`: O3 PGA/classical review plotting.
- `scripts/evaluate_classical_native_resolution.py`: native-GM-input classical inference.

## Tools

- `tools/CalResponseSpectra_src/`: C++ response-spectrum code and Linux batch binary.
