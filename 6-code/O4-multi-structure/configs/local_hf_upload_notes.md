# Local HF Upload Notes for O4 Data

**Date**: 2026-06-04  
**Purpose**: Upload the six O4 IM7 structure HDF5 files to `JasonXF/SeFNO` for server-side training.  
**Status**: Completed on 2026-06-04.

---

## Stable Upload Method

Use the scripted resumable upload instead of a foreground one-off command:

```powershell
$env:HF_TOKEN = "<set locally, do not write into files>"
python 6-code/O4-multi-structure/scripts/upload_o4_hf_data.py --num_workers 4
```

The script:

- reads the token from `HF_TOKEN`;
- clears proxy environment variables by default;
- stages only remote-missing files;
- uses hardlinks instead of copying the 4.75 GB files;
- uses `HfApi.upload_large_folder`, which is resumable.

If the process is interrupted, run the same command again. Already committed files will be skipped, and local upload cache can be reused.

---

## Target Remote Paths

```text
MDOF/knet-250/Data/fno/Blg_F2_6m_IM7_st0.h5
MDOF/knet-250/Data/fno/Blg_F6_18m_IM7_st0.h5
MDOF/knet-250/Data/fno/Blg_F10_30m_IM7_st0.h5
MDOF/knet-250/Data/fno/Blg_F10_30m_IM7_st1.h5
MDOF/knet-250/Data/fno/Blg_F14_42m_IM7_st1.h5
MDOF/knet-250/Data/fno/Blg_F17_52m_IM7_st1.h5
```

Final verified remote state:

```text
MDOF/knet-250/Data/fno/Blg_F2_6m_IM7_st0.h5
MDOF/knet-250/Data/fno/Blg_F6_18m_IM7_st0.h5
MDOF/knet-250/Data/fno/Blg_F10_30m_IM7_st0.h5
MDOF/knet-250/Data/fno/Blg_F10_30m_IM7_st1.h5
MDOF/knet-250/Data/fno/Blg_F14_42m_IM7_st1.h5
MDOF/knet-250/Data/fno/Blg_F17_52m_IM7_st1.h5
```

Upload log:

```text
.codex/hf_upload_logs/o4_hf_upload_20260604_131930.out.log
```
