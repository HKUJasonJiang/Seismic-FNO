# O4 Server Run Notes

**Date**: 2026-06-04  
**Purpose**: Run six FNO-Large multi-structure experiments before returning to O5 FEMA/FRS details.

---

## 1. Environment Check

From the repo root on the server:

```bash
python -m pip install tensorly-torch tqdm h5py scikit-learn
python 6-code/O4-multi-structure/scripts/inspect_o4_data.py \
  --base_data_dir /home/jason/SesimicTransformerData
```

The inspect script should find these six IM7 files:

```text
Blg_F2_6m_IM7_st0.h5
Blg_F6_18m_IM7_st0.h5
Blg_F10_30m_IM7_st0.h5
Blg_F10_30m_IM7_st1.h5
Blg_F14_42m_IM7_st1.h5
Blg_F17_52m_IM7_st1.h5
```

---

## 2. Smoke Run

Run a fast smoke test over all six structures:

```bash
python 6-code/O4-multi-structure/scripts/train_o4_fno_large.py \
  --structures all \
  --base_data_dir /home/jason/SesimicTransformerData \
  --device cuda \
  --epochs 1 \
  --batch_size 1536 \
  --max_batches_per_epoch 2 \
  --max_test_batches 2 \
  --output_dir 6-code/O4-multi-structure/results/smoke_runs
```

If `batch_size=1536` is out of memory, try `1024`, then `512`, and record the largest shared batch size.

---

## 3. Formal Run

After smoke passes:

```bash
python 6-code/O4-multi-structure/scripts/train_o4_fno_large.py \
  --structures all \
  --base_data_dir /home/jason/SesimicTransformerData \
  --device cuda \
  --epochs 50 \
  --batch_size 1536 \
  --output_dir 6-code/O4-multi-structure/results/runs
```

Outputs:

```text
6-code/O4-multi-structure/results/runs/
  F2/
  F6/
  F10/
  FW10/
  FW14/
  FW17/
  o4_training_summary.json
```

Each structure folder contains `config.json`, `training_log.csv`, `checkpoints/`, and `summary.json`.

