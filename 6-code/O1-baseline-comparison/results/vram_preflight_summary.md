# O1 KR1 VRAM Preflight Summary

**Date**: 2026-06-03  
**GPU**: NVIDIA GeForce RTX 5090  
**Dedicated CUDA memory**: 31.84 GB  
**Safety budget**: 90% = 28.66 GB  
**Probe**: one forward + backward + optimizer step per model and batch size  
**Raw CSV**: `vram_preflight.csv`

## Result

Recommended unified batch size for O1 full-data training:

```text
batch_size = 64
```

Reason: `BiLSTM` is the limiting model. It fits comfortably at batch size 64 with 15.55 GB peak allocated memory, but batch size 128 reaches 30.95 GB peak allocated memory, exceeding the 90% dedicated-memory safety budget.

## Peak Allocated CUDA Memory

| Model | bs=16 | bs=32 | bs=64 | bs=128 | bs=256 | Max safe batch |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.24 GB | 0.24 GB | 0.24 GB | 0.24 GB | 0.25 GB | 256 |
| BiLSTM | 4.01 GB | 7.86 GB | 15.55 GB | 30.95 GB | 61.74 GB | 64 |
| ResCNN1D | 0.93 GB | 1.78 GB | 3.44 GB | 6.75 GB | 13.35 GB | 256 |
| UNet1D | 0.82 GB | 1.54 GB | 2.97 GB | 5.80 GB | 11.49 GB | 256 |
| Transformer | 0.68 GB | 1.30 GB | 2.45 GB | 4.82 GB | 9.49 GB | 256 |

## Training Decision

Use a unified batch size of 64 for formal O1 training unless the advisor approves an exception. This keeps the full-data baseline comparison fair and avoids relying on shared GPU memory.

