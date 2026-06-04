# O1 Server Run Notes

**Branch**: `revise`  
**Repo**: `https://github.com/HKUJasonJiang/Seismic-FNO.git`

## 1. Clone

```bash
git clone -b revise https://github.com/HKUJasonJiang/Seismic-FNO.git
cd Seismic-FNO
```

## 2. Environment

Install the usual project requirements plus `tensorly-torch`, which provides `tltorch` for FNO:

```bash
python -m pip install -r requirement.txt
python -m pip install tensorly-torch
```

Use the PyTorch/CUDA build appropriate for the server GPU.

## 3. Data Paths

The scripts default to Jason's Windows paths. On a server, pass explicit paths:

```bash
--base_data_dir /path/to/SesimicTransformerData
--building_dir /path/to/SesimicTransformerData/MDOF/knet-250/Data/fno
```

Required files:

```text
<base_data_dir>/MDOF/All_GMs/GMs_knet_3474_AF_57.h5
<building_dir>/Blg_F6_18m_IM7_st0.h5
```

## 4. Optional VRAM Preflight

```bash
python 6-code/O1-baseline-comparison/scripts/vram_preflight.py \
  --device cuda \
  --batches 16 32 64 128 256
```

Local RTX 5090 result selected:

```text
batch_size = 64
```

## 5. Train Baselines

```bash
python 6-code/O1-baseline-comparison/scripts/train_baselines.py \
  --device cuda \
  --batch_size 64 \
  --epochs 50 \
  --models MLP BiLSTM ResCNN1D UNet1D Transformer \
  --base_data_dir /path/to/SesimicTransformerData \
  --building_dir /path/to/SesimicTransformerData/MDOF/knet-250/Data/fno
```

## 6. Train FNO-Large-bs64

```bash
python 6-code/O1-baseline-comparison/scripts/train_fno_large_bs64.py \
  --device cuda \
  --batch_size 64 \
  --epochs 50 \
  --base_data_dir /path/to/SesimicTransformerData \
  --building_dir /path/to/SesimicTransformerData/MDOF/knet-250/Data/fno
```

## 7. Evaluate and Collect

```bash
python 6-code/O1-baseline-comparison/scripts/evaluate_baselines.py \
  --device cuda \
  --batch_size 64 \
  --base_data_dir /path/to/SesimicTransformerData \
  --building_dir /path/to/SesimicTransformerData/MDOF/knet-250/Data/fno \
  --fno_checkpoint 6-code/O1-baseline-comparison/results/runs/FNO-Large-bs64/checkpoints/FNO-Large-bs64_best.pt

python 6-code/O1-baseline-comparison/scripts/collect_o1_results.py
```

