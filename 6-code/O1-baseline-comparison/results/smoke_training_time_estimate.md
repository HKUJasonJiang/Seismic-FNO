# O1 Smoke Training Time Estimate

**Date**: 2026-06-03  
**Run**: `smoke_runs_bs64_e1`  
**Protocol**: 1 full epoch per model using real HDF5 dataloader, full 57-factor train split, full validation split, `batch_size=64`.

## Dataset

| Split | Samples |
|---|---:|
| Train | 136,800 |
| Validation | 34,200 |
| Test | 27,018 |

## 1-Epoch Smoke Results

The wall time below includes one full train pass and one full validation pass.

| Model | 1 epoch time | Estimated 50 epochs | Peak GPU | Val MSE after 1 epoch | Val R2 after 1 epoch |
|---|---:|---:|---:|---:|---:|
| MLP | 2.74 min | 2.29 h | 0.24 GB | 0.720850 | 0.0889 |
| BiLSTM | 30.52 min | 25.44 h | 15.61 GB | 0.041925 | 0.9470 |
| ResCNN1D | 10.72 min | 8.93 h | 3.49 GB | 0.193820 | 0.7550 |
| UNet1D | 4.02 min | 3.35 h | 3.03 GB | 0.212997 | 0.7308 |
| Transformer | 3.97 min | 3.31 h | 2.51 GB | 0.791506 | -0.0004 |

## Sequential Runtime Estimate

If trained sequentially for 50 epochs each:

```text
MLP + BiLSTM + ResCNN1D + UNet1D + Transformer ≈ 43.3 hours
```

## Interpretation

1. `BiLSTM` is both the VRAM and runtime bottleneck. It fits at `batch_size=64`, but costs about 25.4 hours for 50 epochs.
2. `Transformer` trains quickly but learns almost nothing after 1 epoch. That is not a failure yet; it needs more epochs to judge.
3. `BiLSTM` obtains surprisingly strong validation performance after only 1 epoch. This should be watched carefully: if it remains strong, the paper story should honestly emphasize where FNO differentiates, likely spectral/PFA/generalization/data-efficiency rather than only in-distribution MSE.
4. The formal O1 comparison can be completed in about two days of sequential GPU time, assuming no interruptions.

