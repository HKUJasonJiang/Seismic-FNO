# O1 Code Modification Log

| Timestamp | File | Change | Reason | Compatibility Check |
|---|---|---|---|---|
| 2026-06-03 15:55 | `configs/O1_KR0_protocol.md` | Added O1 full-data baseline protocol and baseline selection rationale | Make KR0 human-reviewable before training | Document only |
| 2026-06-03 15:55 | `scripts/baseline_models.py` | Added MLP, BiLSTM, ResCNN1D, UNet1D, PatchTransformer registry | Implement O1 baseline model set in new `6-code` workspace | Forward pass verified with legacy venv: all output `(2,1,3000)` |
| 2026-06-03 15:55 | `scripts/vram_preflight.py` | Added dedicated VRAM batch-size probe | Enforce fair unified batch-size policy before formal training | `--help` import check passed |
| 2026-06-03 15:55 | `scripts/train_baselines.py` | Added canonical full 57-factor training script with streaming metrics and checkpoints | Train O1 baselines without using old temp workspace | `--help` import check passed |
| 2026-06-03 15:55 | `scripts/evaluate_baselines.py` | Added fixed-test evaluation script for baselines and optional FNO-Large | Produce Table 4 metrics under same split | `--help` import check passed |
| 2026-06-03 15:55 | `../shared/utils/splits.py`, `../shared/utils/metrics.py` | Added canonical split and streaming metric utilities | Share split/metric logic across O1 and later O2/O3 | Python compile check passed |
| 2026-06-03 17:05 | `results/vram_preflight.csv`, `results/vram_preflight_summary.md` | Ran synthetic forward/backward VRAM preflight for all five baselines | Select fair unified batch size before formal training | Recommended `batch_size=64`; BiLSTM is limiting |
| 2026-06-03 18:00 | `results/smoke_runs_bs64_e1/`, `results/smoke_training_time_estimate.md` | Ran one full train+val epoch for each baseline on real HDF5 data | Estimate 50-epoch runtime and catch dataloader/runtime issues | All five models completed; sequential 50-epoch estimate ≈43.3h |
| 2026-06-03 18:05 | `scripts/train_baselines.py` | Added tqdm/batch-level progress output with `--no_progress` fallback | Long training previously had no visible progress between epoch summaries | Pending compile check |
| 2026-06-03 18:25 | `scripts/train_fno_large_bs64.py` | Added fair-comparison FNO-Large training script with `batch_size=64` | Main Table 4 should compare FNO and baselines under same batch size | Pending compile/import check |
| 2026-06-03 18:25 | `scripts/collect_o1_results.py` | Added O1 result collector for plotting-ready CSV/JSON files | Keep training/evaluation metadata in one place for later plotting | Pending compile/import check |
| 2026-06-03 18:35 | `scripts/train_fno_large_bs64.py`, `README.md` | Made FNO import lazy and documented `tensorly-torch` dependency | Current venv lacks `tltorch`; help/config checks should not fail before training | `--help` check pending |
