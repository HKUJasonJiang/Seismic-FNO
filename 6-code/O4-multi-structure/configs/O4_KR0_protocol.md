# O4 KR0 Protocol: Six-Structure FNO-Large Validation

**Version**: v0.1  
**Date**: 2026-06-04  
**Status**: Ready for advisor/server review  
**Paper role**: `§5.4` multi-structure validation and upstream evidence for `§6` FEMA/FRS examples.

---

## 1. Correct Scope

O4 is not a two-case priority run. It is a six-structure validation package.

Target structures:

| ID | Paper Label | Type | Stories | Height | HDF5 |
|---|---|---|---:|---:|---|
| `F2` | F-2 | Frame | 2 | 6 m | `Blg_F2_6m_IM7_st0.h5` |
| `F6` | F-6 | Frame | 6 | 18 m | `Blg_F6_18m_IM7_st0.h5` |
| `F10` | F-10 | Frame | 10 | 30 m | `Blg_F10_30m_IM7_st0.h5` |
| `FW10` | FW-10 | Frame-shear wall | 10 | 30 m | `Blg_F10_30m_IM7_st1.h5` |
| `FW14` | FW-14 | Frame-shear wall | 14 | 42 m | `Blg_F14_42m_IM7_st1.h5` |
| `FW17` | FW-17 | Frame-shear wall | 17 | 52 m | `Blg_F17_52m_IM7_st1.h5` |

All six files were found in:

```text
D:\BaiduNetdiskDownload\SesimicTransformerData\MDOF\knet-250\Data
```

Each target file has:

```text
Acc_Floor_Response: (198018, 3000)
Blg_Damage_State:  (198018, 1)
```

The GM file is:

```text
D:\BaiduNetdiskDownload\SesimicTransformerData\MDOF\All_GMs\GMs_knet_3474_AF_57.h5
Acc_GMs: (198018, 3000)
```

---

## 2. Model Protocol

Use the FNO-Large architecture from the existing paper/hyperparameter search:

| Parameter | Value |
|---|---:|
| `n_modes` | 512 |
| `hidden_channels` | 64 |
| `n_layers` | 8 |
| `domain_padding` | 0.1 |
| `projection_channel_ratio` | 2 |
| input/output channels | 1 / 1 |
| sequence length | 3000 |

Training protocol:

| Item | Default |
|---|---|
| Epochs | 50 |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Scheduler | StepLR, step 20, gamma 0.5 |
| Split | canonical fixed 474-GM test split, 57 factors |
| Seed | 42 |
| Batch size | 1536 for O4 throughput, unless GPU OOM requires fallback |

Reason for batch size:

- O4 is not a baseline fairness comparison, so it does not need the O1 unified `bs=64` rule.
- To keep O4 internally fair, all six structures should use the same batch size if GPU memory permits.
- If the server cannot fit `bs=1536`, test `1024`, then `512`, then record the final shared batch size.

---

## 3. Required Human-Readable Deliverables

Each structure must produce:

1. `config.json`: exact path, split, architecture, optimizer, batch size, seed.
2. `training_log.csv`: epoch-level train/val metrics, PFA metrics, time, GPU peak.
3. `checkpoints/FNO-Large_<structure>_best.pt`: best validation checkpoint.
4. `summary.json`: best validation metrics, test metrics, timing, checkpoint path.

O4 synthesis must produce:

1. `results/compiled/o4_multistructure_summary.csv`
2. Table 6-ready markdown/LaTeX table.
3. Multi-structure convergence figure.
4. Draft `§5.4` text.

---

## 4. Execution Order

This is the real critical path for O4:

```text
KR0 protocol and data paths
  -> KR1 smoke one epoch for all six structures
    -> KR2 run full 50 epochs for all six structures
      -> KR3 evaluate all best checkpoints on the same fixed test set
        -> KR4 compile Table 6 and convergence plot
          -> KR5 use selected outputs downstream for O5 FEMA/FRS
```

O5 details should wait until this O4 result package is underway or complete.

