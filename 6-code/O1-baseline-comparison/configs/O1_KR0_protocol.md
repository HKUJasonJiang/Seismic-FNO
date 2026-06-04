# O1 KR0 Protocol: Full-Data Baseline Comparison

**Version**: v0.1  
**Date**: 2026-06-03  
**Paper Sections**: `§2.3`, `§4.3`  

## Goal

Provide a fair full-data comparison between FNO-Large and five neural surrogate baselines.

The main comparison should include a newly trained **FNO-Large-bs64** model. The legacy high-throughput FNO-Large checkpoint can be reported as a production/reference configuration, but the main Table 4 should use the bs=64 FNO run for strict training-protocol fairness.

## Baseline Set

| Model | Keep? | Rationale |
|---|---|---|
| MLP | Yes | Simple global sequence mapping; important sanity baseline. |
| BiLSTM | Yes | Classic temporal sequence baseline expected by reviewers. |
| ResCNN1D / TCN-style CNN | Yes | Local/dilated convolution baseline; should be described as 1D-CNN/TCN family. |
| UNet1D | Yes | Multi-scale encoder-decoder baseline for time histories. |
| Patch Transformer | Yes | Attention-based modern sequence baseline. |

Not included in main O1:

- Gaussian process: not practical for 198k samples x 3000-step output without a separate approximation framework.
- ROM/modal baseline: useful but changes the comparison from neural surrogate to reduced-order structural mechanics; consider only if reviewers demand it.
- DeepONet: relevant, but requires branch/trunk design choices and would become a separate method paper comparison. Keep as optional extension after O1 if needed.

## Fairness Rules

1. Use the corrected fixed split:
   - Train: 2,400 GMs x 57 = 136,800 samples
   - Validation: 600 GMs x 57 = 34,200 samples
   - Test: 474 GMs x 57 = 27,018 samples
2. Use all 57 factors for O1.
3. Use AdamW, MSE, StepLR(20, 0.5), seed 42.
4. Before formal training, run VRAM preflight on all models and select the largest unified batch size that fits dedicated GPU memory for all models.
5. Report any exception to the unified batch-size rule.

## KR Deliverables

| KR | Deliverable |
|---|---|
| KR1 | VRAM table and selected batch size |
| KR2-KR6 | One training log summary and one test row per baseline |
| KR7 | FNO-Large-bs64 training log summary and aligned test row |
| KR8 | Compiled plotting-ready CSV/JSON result files |
| KR9 | Table 4, representative overlay, PFA/spectral figure, draft `§4.3` text |

## Plotting-Ready Result Location

All model-level metadata should be collected into:

```text
6-code/O1-baseline-comparison/results/compiled/
```

Expected files:

- `training_runs.csv`
- `training_runs.json`
- `table4_metrics.csv`
- `manifest.json`
