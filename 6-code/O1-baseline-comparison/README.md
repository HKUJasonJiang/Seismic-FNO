# O1 Baseline Comparison

Paper role: `§2.3` and `§4.3`.

Purpose: run a fair full-data comparison between FNO-Large and baseline surrogate models using the corrected fixed split and all 57 factors.

Key rules:

1. Run VRAM preflight before formal training.
2. Select the largest unified batch size that all models can run under dedicated GPU memory.
3. Use the corrected fixed 474-GM test split in `../shared/split_indices/`.
4. Record training time, dedicated VRAM peak, best checkpoint, and test metrics for every model.

Expected paper deliverables:

- Table 4: model comparison metrics.
- Representative time-history overlay.
- PFA and spectral comparison figures.
- Draft text for `§4.3`.

## FNO-Large Fair-Comparison Run

The main Table 4 should use a newly trained `FNO-Large-bs64` model:

Dependency note: current `neuralop` imports `tltorch`, provided by `tensorly-torch`. If the active venv does not have it, install before running FNO training:

```powershell
.\workbuddy\legacy-code\Seismic-FNO-temp\.venv\Scripts\python.exe -m pip install tensorly-torch
```

```powershell
.\workbuddy\legacy-code\Seismic-FNO-temp\.venv\Scripts\python.exe .\6-code\O1-baseline-comparison\scripts\train_fno_large_bs64.py --device cuda --batch_size 64 --epochs 50
```

Rationale: legacy FNO-Large was trained with a much larger batch size. For strict fairness, O1 needs a same-batch FNO-Large run alongside the five baselines.

## Baseline Choice

The current main-paper baseline set is:

| Model | Role |
|---|---|
| `MLP` | Simple global sequence mapping sanity baseline |
| `BiLSTM` | Classic recurrent temporal baseline |
| `ResCNN1D` | Dilated 1D-CNN / TCN-style local temporal baseline |
| `UNet1D` | Multi-scale convolutional encoder-decoder |
| `Transformer` | Attention-based sequence baseline |

I do not recommend replacing these in the main O1 comparison now. GP is not practical at this output scale, ROM changes the problem class, and DeepONet would require a separate branch/trunk design study. They can be discussed as optional extensions if reviewers demand them.

## Standard Commands

Run VRAM preflight first:

```powershell
cd D:\BaiduSyncdisk\Paper\6-SeFNO
.\workbuddy\legacy-code\Seismic-FNO-temp\.venv\Scripts\python.exe .\6-code\O1-baseline-comparison\scripts\vram_preflight.py --device cuda --batches 16 32 64 128 256
```

Train after the unified batch size is selected:

```powershell
.\workbuddy\legacy-code\Seismic-FNO-temp\.venv\Scripts\python.exe .\6-code\O1-baseline-comparison\scripts\train_baselines.py --device cuda --batch_size <SELECTED_BS> --epochs 50 --models MLP BiLSTM ResCNN1D UNet1D Transformer
```

Evaluate:

```powershell
.\workbuddy\legacy-code\Seismic-FNO-temp\.venv\Scripts\python.exe .\6-code\O1-baseline-comparison\scripts\evaluate_baselines.py --device cuda --batch_size <SELECTED_BS> --fno_checkpoint <FNO_LARGE_CHECKPOINT>
```

Collect plotting-ready run metadata:

```powershell
.\workbuddy\legacy-code\Seismic-FNO-temp\.venv\Scripts\python.exe .\6-code\O1-baseline-comparison\scripts\collect_o1_results.py
```

The current command examples use the archived venv only because it already contains the RTX 5090-compatible PyTorch build. A clean active venv can replace it later.
