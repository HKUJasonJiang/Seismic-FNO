# Remote Sync Confirmation Plan v0.1

Date: 2026-06-09

Purpose: reconcile the local paper-control workspace with experiment code and artifacts produced on XG-Boost-1 and XG-Boost-2 before continuing O5.

## Current Remote Inventory

| Machine | SSH alias checked | Repo path | Branch / commit | Objective state | Artifact size | Notes |
|---|---|---|---|---|---:|---|
| XG-Boost-1 | `XG-Boost-1-jason` | `/home/jason/Seismic-FNO` | `revise` / `e83a165` | O4 completed | O4 results 3.5 GB; `6-code/data` 540 MB | One meaningful code change in O4 trainer plus runtime pollution from `__pycache__`; O2 not started here yet. |
| XG-Boost-2 | `XG-Boost-2-jason` | `/data/home/jason/Seismic-FNO` | `revise` / `02a023a` | O1 and O3 completed; O2 partial baseline-only package exists | O1 854 MB; O3 1.8 GB; O2 partial 294 MB | Has new O1/O2/O3 code and reports; `XG-Boost-2-user` SSH is not accessible from this local machine. |

## Objective-Level Status

| Objective | Where it ran | Observed evidence | Current interpretation | Sync priority |
|---|---|---|---|---|
| O1 baseline comparison | XG-Boost-2 | `O1_baseline_report.*`, `o1_report_summary.csv`, metric figures, checkpoints | Completed. Important result: on full F-6 data, BiLSTM ranks best, Transformer second, FNO-Large third. This changes the paper story. | Highest, because XG-Boost-2 needs to be freed. |
| O2 data efficiency | XG-Boost-2 partial; planned for XG-Boost-1 | `E-Base` rows from O1 plus `E-AC80` for FNO-Large and BiLSTM only | Not complete. Existing result is a partial/early package, not the final data-efficiency matrix. | After O1/O3 artifact rescue, run full O2 on XG-Boost-1. |
| O3 generalization | XG-Boost-2 | PGA OOD review figures, sampling OOD runs, classical GM analysis, response spectra files | Completed enough to sync and review. Needs local consolidation before dashboard can mark done. | Highest, because XG-Boost-2 needs to be freed. |
| O4 multi-structure | XG-Boost-1 | Six FNO-Large structure runs; compiled summary table and figures | Completed. R2 range roughly 0.864-0.926 across F/FW structures; peak GPU about 48.9 GB. | High, because O5 depends on O4 artifacts. |
| O5 FEMA P58 / FRS | Local | Existing local work only | Should resume only after O4 evidence and O5-ready arrays/checkpoints are synced. | Starts after O4 sync package is verified locally. |

## Key Results Already Observed

### O1

Full-data, batch size 64 comparison:

| Model | Test MSE | Test R2 | PFA RMSE (g) | Train time (h) | Peak GPU (GB) |
|---|---:|---:|---:|---:|---:|
| BiLSTM | 0.0099 | 0.9872 | 0.0182 | 7.51 | 15.61 |
| Transformer | 0.0214 | 0.9725 | 0.0231 | 2.34 | 2.54 |
| FNO-Large | 0.0449 | 0.9423 | 0.0330 | 2.16 | 3.30 |
| ResCNN1D | 0.1481 | 0.8096 | 0.0547 | 6.29 | 3.49 |
| UNet1D | 0.2157 | 0.7227 | 0.1063 | 2.27 | 3.03 |
| MLP | 0.7420 | 0.0462 | 0.2464 | 1.59 | 0.24 |

Implication: O1 cannot be narrated as FNO-Large winning the full-data single-structure benchmark. The stronger paper story should shift toward where FNO is valuable: multi-structure consistency, reduced data, engineering/FEMA workflow, and possibly speed/parameterized operators.

### O4

Six-structure FNO-Large result:

| Structure | Type | Test R2 | Test RMSE | PFA RMSE (g) | Peak GPU (GB) |
|---|---|---:|---:|---:|---:|
| F-10 | Frame | 0.9107 | 0.1972 | 0.0283 | 48.92 |
| F-2 | Frame | 0.9258 | 0.2459 | 0.0355 | 48.92 |
| F-6 | Frame | 0.8959 | 0.2845 | 0.0392 | 48.92 |
| FW-10 | Frame-shear wall | 0.8636 | 0.4057 | 0.0734 | 48.92 |
| FW-14 | Frame-shear wall | 0.8920 | 0.3139 | 0.0549 | 48.92 |
| FW-17 | Frame-shear wall | 0.8888 | 0.3054 | 0.0566 | 48.92 |

Implication: O4 is the current bridge from model training to O5. Local O5 should consume an explicit O4 package instead of manually depending on scattered server run folders.

## Confirmation Table: Proposed Next Actions

| Step | Action | Source | Target | What will happen | Expected result |
|---:|---|---|---|---|---|
| 1 | Rescue XG-Boost-2 code changes | XG-Boost-2 repo | GitHub `revise` or a temporary sync branch | Review/commit only real O1/O2/O3 code and docs; exclude `__pycache__`, pids, heavy results, raw data. | XG-Boost-2 code changes become reproducible before the machine is reused. |
| 2 | Sync XG-Boost-2 evidence artifacts | O1/O3/O2 partial results | Local `6-code/.../results/remote_sync/...` and/or HF artifact area | Pull reports, csv/json summaries, figures, logs, run scripts; defer checkpoints unless needed. | Local dashboard can mark O1 and O3 as synced; O2 remains partial. |
| 3 | Preserve XG-Boost-2 heavy checkpoints | O1/O3 checkpoints | HF dataset or server-side archive manifest | Upload or index checkpoint paths with hashes/sizes; download locally only if later analysis needs them. | XG-Boost-2 can be freed without losing reproducibility. |
| 4 | Rescue XG-Boost-1 O4 code change | XG-Boost-1 repo | GitHub `revise` or temporary sync branch | Extract O4 trainer improvements, ignore runtime pollution and large results. | O4 training script used for final runs is preserved. |
| 5 | Sync O4 evidence and O5-ready package | XG-Boost-1 O4 results | Local O4 results plus `O5-fema-p58/input/o4_ready/` | Pull compiled summary, training logs, figures, test metrics; decide whether to pull six best checkpoints or export predictions/response arrays. | O5 can run locally from a stable input package. |
| 6 | Run full O2 on XG-Boost-1 | XG-Boost-1 | XG-Boost-1 results then sync protocol | Use the completed O1 baselines and planned reduced-data grid; include all required models. | O2 final matrix replaces the partial E-AC80-only evidence. |
| 7 | Update project dashboard | Local `.codex` docs | `.codex/progress-dashboard.md` and `project-manage.md` | Mark objectives only after evidence is verified locally. | This local thread becomes the single reliable project panel again. |

## Sync Rules

- GitHub gets code, configs, docs, lightweight reports, and generated tables/figures only when useful.
- HF or server archive gets checkpoints, prediction arrays, and large response-spectrum/O5 packages.
- Local machine gets all paper evidence and O5-required inputs, not necessarily every checkpoint.
- Runtime byproducts such as `__pycache__`, `.pid`, temporary monitor scripts, and failed/duplicate smoke checkpoints should not be treated as final evidence.
- O2 must remain marked partial until reduced-data runs exist for the planned model/data-ratio matrix.

## Open Decisions Before Execution

1. Whether to sync heavy checkpoints locally or keep them on HF/server with manifests.
2. Whether O4 should export predictions/PFA/FRS arrays for O5, or whether local O5 should load six checkpoints and recompute.
3. Whether server code should be merged directly into `revise` or first pushed to temporary branches such as `sync-xgb2-o1-o3` and `sync-xgb1-o4`.

## v0.2 Full Sync Plan Before Running O2

Decision update from user: use `JasonXF/SeFNO` as the single Hugging Face dataset repo for paper synchronization. Do not use `JasonXF/SeFNO-Paper` for now.

Sensitive-token rule: HF tokens must never be written to this file, GitHub, scripts, or persistent shell profiles. Use only temporary environment variables inside the upload command/session.

### Target State

After synchronization:

- GitHub `HKUJasonJiang/Seismic-FNO` contains the latest code, configs, project docs, and lightweight evidence needed to reproduce/inspect O1/O3/O4 and current partial O2.
- Hugging Face `JasonXF/SeFNO` contains the heavy experiment artifacts: checkpoints, large outputs, O4/O5-ready packages, and machine-readable manifests.
- Local workspace contains all paper-facing evidence and a verified local mirror of the latest lightweight results.
- Existing local files are protected by timestamped backups before any merge/copy operation.
- XG-Boost-1 is then clear to start full O2.

### Remote Branch Strategy

| Source | Branch | Purpose | Merge policy |
|---|---|---|---|
| XG-Boost-2 | `sync/xgb2-o1-o3-o2partial` | O1/O3 completed code and evidence generation scripts; current O2 partial scripts/reports | Push first, inspect locally, then merge into `revise` only after backup |
| XG-Boost-1 | `sync/xgb1-o4` | O4 trainer changes and O4 evidence generation state | Push first, inspect locally, then merge into `revise` only after backup |
| XG-Boost-1 later | `sync/xgb1-o2-final` | Full O2 after it is run | Same procedure after O2 completes |

### Hugging Face Layout

Use this layout under `JasonXF/SeFNO`:

```text
experiments/
  O1_baseline/
    2026-06-09_xgb2_full_bs64/
      manifest.json
      evidence/
      checkpoints/
  O2_data_efficiency/
    2026-06-09_xgb2_partial_EAC80/
      manifest.json
      evidence/
      checkpoints/
    2026-06-xx_xgb1_final/
      manifest.json
      evidence/
      checkpoints/
  O3_generalization/
    2026-06-09_xgb2_pga_sampling_gmood/
      manifest.json
      evidence/
      checkpoints/
      large_inputs/
  O4_multi_structure/
    2026-06-09_xgb1_fno_large_6struct/
      manifest.json
      evidence/
      checkpoints/
      o5_ready/
```

### Local Backup Policy

Before pulling/merging/copying any remote results into local `6-code`:

```text
.codex/backups/remote-sync-YYYYMMDD-HHMMSS/
  git_status_before.txt
  changed_local_files/
  current_codex_docs/
  current_result_indexes/
```

The backup must be made before:

- merging sync branches into local `revise`
- copying server evidence into local results folders
- updating `.codex/progress-dashboard.md`

### Sync Order

| Step | Action | Safety check | Expected result |
|---:|---|---|---|
| 1 | Create local backup snapshot | Verify backup files exist | Local state can be restored if the merge/copy strategy is wrong |
| 2 | On XG-Boost-2, create `sync/xgb2-o1-o3-o2partial` | Exclude checkpoints, raw data, `__pycache__`, pids | O1/O3/O2-partial code and lightweight evidence are on GitHub |
| 3 | Upload XG-Boost-2 heavy artifacts to `JasonXF/SeFNO` | Prefer direct HF login on XG2; fallback is transfer to XG1 then upload | O1/O3 checkpoints and large O3/O2 partial files are preserved on HF |
| 4 | On XG-Boost-1, create `sync/xgb1-o4` | Exclude `__pycache__`, smoke duplicate checkpoints unless intentionally archived | O4 code/evidence state is on GitHub |
| 5 | Upload XG-Boost-1 O4 heavy artifacts to `JasonXF/SeFNO` | Verify manifests and file counts on HF | Six O4 best checkpoints and O5-ready inputs are preserved |
| 6 | Local fetch and inspect sync branches | Review `git diff` before merge | No accidental deletion or huge-file Git pollution |
| 7 | Merge approved code/evidence into local `revise` | Confirm local backup exists | Local code and lightweight evidence become latest |
| 8 | Download/pull HF evidence needed locally | Verify manifest checksums when possible | Local has all results needed for review and O5 |
| 9 | Update `.codex/progress-dashboard.md` | Only mark synced objectives after files exist locally | Dashboard becomes truthful again |
| 10 | Start full O2 on XG-Boost-1 | Only after O1/O3/O4 sync is verified | Server 1 proceeds with O2 from a clean known state |

### Result Classification

GitHub:

- source code
- configs
- project docs
- report `.md/.tex`
- small `.csv/.json`
- selected figures and PDFs
- artifact manifests

Hugging Face:

- `.pt` checkpoints
- large arrays
- O4/O5-ready packages
- large response-spectrum or ground-motion derived packages
- full heavy result archives

Local:

- all GitHub material
- all paper-facing evidence
- all O5-required inputs
- selected checkpoints only if needed to recompute O5

## v0.3 Sync Completion Record

Completed on 2026-06-09.

### Backups Created

| Backup | Purpose |
|---|---|
| `sync-backups/remote-sync-20260609-144342` | Main local backup before sync; includes `.codex` docs, `CODEX.md`, full local O5 F-6 sample, and result index |
| `sync-backups/remote-sync-20260609-144951-pre-active-o1-o4-copy` | Full local active O1-O4 backup before copying merged lightweight evidence from clean clone |
| `sync-backups/remote-bundles/` | Local copies of remote Git bundles from XG-Boost-1 and XG-Boost-2 |

### GitHub Sync

| Ref | Commit | Purpose |
|---|---|---|
| `revise` | `dc5de70` | Merged latest O1/O2 partial/O3/O4 code and lightweight evidence |
| `sync/xgb2-o1-o3-o2partial` | `df66472` | Audit branch for XG-Boost-2 O1/O3/O2 partial sync |
| `sync/xgb1-o4` | `1911060` | Audit branch for XG-Boost-1 O4 sync |

### Hugging Face Sync

All target manifests returned HTTP 200 after upload.

| Objective | HF path | HF commit |
|---|---|---|
| O1 baseline | `experiments/O1_baseline/2026-06-09_xgb2_full_bs64/manifest.json` | `dec99c4b07d8cf1176fec049da5d6bb7054ec05d` |
| O2 data efficiency partial | `experiments/O2_data_efficiency/2026-06-09_xgb2_partial_EAC80/manifest.json` | `db1dd038c769568ba21618bd91595c6c5ae592bd` |
| O3 generalization | `experiments/O3_generalization/2026-06-09_xgb2_pga_sampling_gmood/manifest.json` | `f2d1b88aa84ff3cfd405be7c3dac32c19cfea8c2` |
| O4 multi-structure | `experiments/O4_multi_structure/2026-06-09_xgb1_fno_large_6struct/manifest.json` | `2ae19ce72b74cec44009d486732bc52e8a2aee8b` |
| O5 local F-6 sample | `experiments/O5_fema_p58/2026-06-09_local_F6_sample/manifest.json` | `df5ed30ac6766021cd2bd273dd2638403b70fdab` |

### Local Sync Verification

Verified local active files:

- `6-code/O1-baseline-comparison/results/report/O1_baseline_report.pdf`
- `6-code/O2-data-efficiency/results/report/O2_data_efficiency_report.md`
- `6-code/O3-generalization/results/figures_o3_review/pga_ood_by_scale_mse_r2.png`
- `6-code/O4-multi-structure/results/compiled/o4_multistructure_summary.csv`
- `6-code/O5-fema-p58/results/f6_verified/f6_p58_verified_results.json`

### Remaining Before Full O2

- XG-Boost-1 full O2 has been started from a clean export of `origin/revise` at `/home/jason/Seismic-FNO-O2-20260609`.
- O2 must remain `partial` until the planned reduced-data matrix is completed and uploaded as `experiments/O2_data_efficiency/2026-06-xx_xgb1_final/`.
- The HF write token was used only as a temporary command environment variable for uploads; it should be rotated after this sync because it appeared in chat context.

O2 run monitor:

- PID: `189895`
- Log: `/home/jason/Seismic-FNO-O2-20260609/6-code/O2-data-efficiency/results/logs/o2_xgb1_full_sequential_20260609_070936.log`
- GPU: single RTX PRO 6000 Blackwell, queue order `FNO-Large -> Transformer -> BiLSTM`
- First verified active run: `FNO-Large E-AC80`
