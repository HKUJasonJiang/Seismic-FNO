# Server Takeover: Finish O4 Multi-Structure Validation

**Date**: 2026-06-04  
**Objective**: Complete O4 six-structure FNO-Large validation without further back-and-forth.

---

## 1. Pull Code

```bash
git clone -b revise https://github.com/HKUJasonJiang/Seismic-FNO.git
cd Seismic-FNO
```

If already cloned:

```bash
git fetch origin
git checkout revise
git pull origin revise
```

Expected O4 code commit:

```text
6a98315 Add O4 six-structure FNO training workflow
```

---

## 2. Prepare Python Environment

Use the project/server virtual environment. Install missing dependencies:

```bash
python -m pip install -U huggingface_hub hf_xet tensorly-torch tqdm h5py scikit-learn
```

The FNO script requires `tensorly-torch` because `neuralop` imports `tltorch`.

---

## 3. Download O4 Data from Hugging Face

Set a local Hugging Face token on the server if the dataset requires authentication:

```bash
export HF_TOKEN="<set on server, do not commit>"
```

Download only the files needed for O4:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="JasonXF/SeFNO",
    repo_type="dataset",
    allow_patterns=[
        "MDOF/All_GMs/GMs_knet_3474_AF_57.h5",
        "MDOF/knet-250/Data/fno/Blg_F2_6m_IM7_st0.h5",
        "MDOF/knet-250/Data/fno/Blg_F6_18m_IM7_st0.h5",
        "MDOF/knet-250/Data/fno/Blg_F10_30m_IM7_st0.h5",
        "MDOF/knet-250/Data/fno/Blg_F10_30m_IM7_st1.h5",
        "MDOF/knet-250/Data/fno/Blg_F14_42m_IM7_st1.h5",
        "MDOF/knet-250/Data/fno/Blg_F17_52m_IM7_st1.h5",
    ],
    local_dir="/home/jason/SesimicTransformerData",
)
PY
```

After download, these files should exist:

```text
/home/jason/SesimicTransformerData/MDOF/All_GMs/GMs_knet_3474_AF_57.h5
/home/jason/SesimicTransformerData/MDOF/knet-250/Data/fno/Blg_F2_6m_IM7_st0.h5
/home/jason/SesimicTransformerData/MDOF/knet-250/Data/fno/Blg_F6_18m_IM7_st0.h5
/home/jason/SesimicTransformerData/MDOF/knet-250/Data/fno/Blg_F10_30m_IM7_st0.h5
/home/jason/SesimicTransformerData/MDOF/knet-250/Data/fno/Blg_F10_30m_IM7_st1.h5
/home/jason/SesimicTransformerData/MDOF/knet-250/Data/fno/Blg_F14_42m_IM7_st1.h5
/home/jason/SesimicTransformerData/MDOF/knet-250/Data/fno/Blg_F17_52m_IM7_st1.h5
```

---

## 4. Verify Data Schema

```bash
python 6-code/O4-multi-structure/scripts/inspect_o4_data.py \
  --base_data_dir /home/jason/SesimicTransformerData
```

Expected for every structure:

```text
Acc_Floor_Response: shape=(198018, 3000)
Blg_Damage_State:  shape=(198018, 1)
```

---

## 5. Smoke Run

Run a short six-structure smoke test:

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

If CUDA OOM occurs, retry shared batch size in this order:

```text
1024 -> 512 -> 256
```

Record the final shared batch size in `6-code/O4-multi-structure/results/smoke_runs/batchsize_decision.txt`.

---

## 6. Full O4 Run

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

Use the smoke-approved batch size if it is lower than 1536.

---

## 7. Compile Results

```bash
python 6-code/O4-multi-structure/scripts/collect_o4_results.py \
  --runs_dir 6-code/O4-multi-structure/results/runs \
  --output_dir 6-code/O4-multi-structure/results/compiled
```

Expected outputs:

```text
6-code/O4-multi-structure/results/compiled/o4_multistructure_summary.csv
6-code/O4-multi-structure/results/compiled/o4_multistructure_table.md
```

---

## 8. Completion Criteria

O4 is complete when all six structures have:

- `config.json`
- `training_log.csv`
- `checkpoints/FNO-Large_<structure>_best.pt`
- `summary.json`
- compiled summary CSV/table

Do not proceed to detailed O5 FEMA/FRS scenario conclusions until O4 is complete or at least all six runs have valid test summaries.

