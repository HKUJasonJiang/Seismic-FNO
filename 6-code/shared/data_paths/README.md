# Data Path Notes

**Date**: 2026-06-03  
**Purpose**: canonical path notes for active experiments.

## Ground Motion Data

```text
D:\BaiduNetdiskDownload\SesimicTransformerData\MDOF\All_GMs\GMs_knet_3474_AF_57.h5
```

## Building HDF5 Data

Main data directory:

```text
D:\BaiduNetdiskDownload\SesimicTransformerData\MDOF\knet-250\Data
```

Current active F-6 training folder used by old scripts:

```text
D:\BaiduNetdiskDownload\SesimicTransformerData\MDOF\knet-250\Data\fno
```

As of the handoff, `Data\fno` contains only:

```text
Blg_F6_18m_IM7_st0.h5
```

For O4 multi-structure work, do not rely on `Data\fno` implicitly. Use an explicit data-path configuration or a controlled copy/symlink workflow.

## Corrected Split

The corrected fixed split is stored in:

```text
6-code\shared\split_indices
```

These files were copied from the legacy corrected split:

- `dataset_indices.pkl`
- `dataset_indices.txt`
- `split_info.txt`

Canonical split:

| Split | GMs | Samples |
|---|---:|---:|
| Train | 2,400 | 136,800 |
| Validation | 600 | 34,200 |
| Test | 474 | 27,018 |

## Clean Upstream Code

Clean GitHub clone:

```text
6-code\shared\upstream\Seismic-FNO-clean
```

Legacy modified code archive:

```text
workbuddy\legacy-code\Seismic-FNO-temp
```

