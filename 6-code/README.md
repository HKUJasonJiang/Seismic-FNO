# 6-code Active Experiment Workspace

This folder is the active code and experiment workspace for the Engineering Structures resubmission.

Rules:

1. New work is organized by Objective, not by file type.
2. Each Objective folder must keep its own `README.md`, `code-modify.md`, configs, scripts, logs, results, and paper-facing figures/tables.
3. Legacy code is archived under `workbuddy/`; do not continue expanding old temporary folders.
4. Original paper assets in `1-manuscript` to `5-submission` are source material and should not be overwritten.

Objective folders:

| Folder | Purpose |
|---|---|
| `O1-baseline-comparison/` | Full 57-factor fair comparison: FNO-Large vs. baselines |
| `O2-data-efficiency/` | Reduced-factor data-efficiency comparison for FNO and baselines |
| `O3-generalization/` | OOD descriptors, zero-shot evaluation, PGA extrapolation |
| `O4-multi-structure/` | FNO-Large validation across selected structures |
| `O5-fema-p58/` | FEMA P-58 parameter verification and NSC damage assessment |
| `O6-manuscript-assembly/` | Section drafts, final tables/figures, compile logs |
| `shared/` | Split indices, common utilities, clean upstream source |

