# SeFNO ES Resubmission Progress Dashboard

**Version**: v0.2  
**Last Updated**: 2026-06-09  
**Primary Management File**: `D:\BaiduSyncdisk\Paper\6-SeFNO\.codex\project-manage.md`  
**Active Code Root**: `D:\BaiduSyncdisk\Paper\6-SeFNO\6-code`

---

## 0. How to Use This Dashboard

This is the single progress panel for resuming work after interruptions or unrelated conversations.

Status legend:

| Status | Meaning |
|---|---|
| `Not started` | No meaningful execution yet |
| `Planned` | KR protocol/design exists, but work not run |
| `In progress` | Work has started but deliverable is incomplete |
| `Done` | Human-reviewable KR deliverable exists |
| `Blocked` | Cannot proceed without dependency/user/external change |
| `Prototype only` | A draft/prototype exists but is not publication-safe |

Update rule:

- After running any KR, update the relevant Objective table.
- After generating a result file, add its path to the "Evidence" column.
- After advisor changes strategy, update `project-manage.md` first, then this dashboard.

---

## 2026-06-09 Remote Sync Snapshot

| Stream | Current state | Local evidence | GitHub | Hugging Face |
|---|---|---|---|---|
| O1 baseline | Synced from XG-Boost-2; full-data bs64 run complete | `6-code/O1-baseline-comparison/results/report/O1_baseline_report.pdf`; `results/report/o1_report_summary.csv` | `revise` at `dc5de70`; audit branch `sync/xgb2-o1-o3-o2partial` at `df66472` | `JasonXF/SeFNO/experiments/O1_baseline/2026-06-09_xgb2_full_bs64/manifest.json` |
| O2 data efficiency | Partial only; XG-Boost-2 has E-AC80 package and base rows, not final matrix | `6-code/O2-data-efficiency/results/report/O2_data_efficiency_report.md` | same as above | `JasonXF/SeFNO/experiments/O2_data_efficiency/2026-06-09_xgb2_partial_EAC80/manifest.json` |
| O3 generalization | Synced from XG-Boost-2 | `6-code/O3-generalization/results/figures_o3_review/`; classical/GM OOD result folders | `revise` at `dc5de70`; audit branch `sync/xgb2-o1-o3-o2partial` | `JasonXF/SeFNO/experiments/O3_generalization/2026-06-09_xgb2_pga_sampling_gmood/manifest.json` |
| O4 multi-structure | Synced from XG-Boost-1; six-structure FNO-Large run complete | `6-code/O4-multi-structure/results/compiled/o4_multistructure_summary.csv` | `revise` at `dc5de70`; audit branch `sync/xgb1-o4` at `1911060` | `JasonXF/SeFNO/experiments/O4_multi_structure/2026-06-09_xgb1_fno_large_6struct/manifest.json` |
| O5 FEMA P-58 | Local F-6 prototype/sample preserved and synced | `6-code/O5-fema-p58/results/f6_verified/f6_p58_verified_results.json` | local active workspace; not yet folded into a dedicated O5 Git sync branch | `JasonXF/SeFNO/experiments/O5_fema_p58/2026-06-09_local_F6_sample/manifest.json` |

Backup checkpoints before sync:

- Main O5/docs backup: `sync-backups/remote-sync-20260609-144342`
- Pre-active-copy O1-O4 backup: `sync-backups/remote-sync-20260609-144951-pre-active-o1-o4-copy`
- Remote Git bundles: `sync-backups/remote-bundles/`

Critical reading note: O1 full-data baseline does **not** show FNO-Large as the best single-structure full-data model. The current O1 ranking is BiLSTM first, Transformer second, FNO-Large third by test MSE/R2. The paper story should therefore use O1 as a fair baseline calibration and shift FNO claims toward multi-structure, data-efficiency, and engineering/FEMA usefulness.

O2 full run started on XG-Boost-1 after sync:

- Worktree/export: `/home/jason/Seismic-FNO-O2-20260609`
- PID: `189895`
- Log: `/home/jason/Seismic-FNO-O2-20260609/6-code/O2-data-efficiency/results/logs/o2_xgb1_full_sequential_20260609_070936.log`
- Queue order: `FNO-Large -> Transformer -> BiLSTM`
- Current first run at launch verification: `FNO-Large E-AC80`

---

## 1. Current Big Picture

The revised Engineering Structures paper is managed as:

```text
§4: Full-data fair competition
§5: Data efficiency + generalization + multi-structure validation
§6: FEMA P-58 engineering value
```

Current execution priority:

```text
server-side O1 full-data baseline comparison
  -> baseline/FNO-Large-bs64 training and same-test evaluation

local critical path
  -> O4 six-structure path/config check
  -> O4 six-structure smoke run
  -> O4 six-structure 50-epoch training/evaluation
  -> O4 Table 6-ready summary
  -> O5 FEMA/FRS scenario planning using verified O4 outputs

local O5 template path
  -> F-6-only FEMA P-58 prototype template
  -> lock result schema, table shape, and SVG figure style
  -> later replace prototype parameters and add FNO/O4 outputs
```

Critical current decision already made:

- O1 uses unified `batch_size=64`.
- FNO-Large should also be trained as `FNO-Large-bs64` for the main fair Table 4 comparison.
- Legacy high-throughput FNO-Large can be used as supplementary/production reference, not the only main comparator.
- O5 detailed scenario planning should not run ahead of the full O4 six-structure package. FEMA source review can continue in parallel, but O4 is the immediate local critical path.
- Advisor update on 2026-06-04: O5 may proceed with an F-6-only template before multi-structure is available, as long as it is clearly marked prototype-only and used to prepare the P58 workflow/layout rather than final claims.

---

## 2. Active Markdown Map

## Control Layer

| File | Purpose |
|---|---|
| `.codex/README.md` | Navigation for active management docs |
| `.codex/project-manage.md` | Master story, OKR map, critical path, directory rules |
| `.codex/progress-dashboard.md` | This progress panel |

## Code/Objective Layer

| Folder | Main MD Files | Purpose |
|---|---|---|
| `6-code/` | `README.md` | Active experiment workspace overview |
| `6-code/O1-baseline-comparison/` | `README.md`, `code-modify.md`, `configs/O1_KR0_protocol.md` | Full-data FNO vs baseline comparison |
| `6-code/O2-data-efficiency/` | `README.md`, `code-modify.md` | Reduced-factor data efficiency for FNO and baselines |
| `6-code/O3-generalization/` | `README.md`, `code-modify.md` | OOD, zero-shot, PGA extrapolation |
| `6-code/O4-multi-structure/` | `README.md`, `code-modify.md` | Multi-structure FNO-Large validation |
| `6-code/O5-fema-p58/` | `README.md`, `code-modify.md` | FEMA P-58 verification and damage assessment |
| `6-code/O6-manuscript-assembly/` | `README.md`, `code-modify.md` | Manuscript integration |
| `6-code/shared/` | `README.md`, `data_paths/README.md` | Shared paths, splits, upstream code |

## Legacy Layer

| Folder | Purpose |
|---|---|
| `workbuddy/` | Archived old AI work, old temp code, old prototype results |
| `1-manuscript/` to `5-submission/` | Original manuscript/data/fig/ref/submission assets; preserve as source material |

---

## 3. Objective Progress

## O0: Project Story and Management Layer

| KR | Deliverable | Status | Evidence |
|---|---|---|---|
| KR0 | Advisor reviews story and section map | Done | `.codex/project-manage.md` v0.2 |
| KR1 | Active docs moved from `.workbuddy` to `.codex` | Done | `.codex/README.md`, `.codex/project-manage.md` |
| KR2 | Active code workspace organized by Objective | Done | `6-code/README.md`, `6-code/O*/README.md` |
| KR3 | Single progress dashboard created | Done | `.codex/progress-dashboard.md` |

Next action: keep this dashboard updated after every KR.

---

## O1: Full-Data Baseline Comparison

**Paper role**: `§2.3`, `§4.3`  
**Active folder**: `6-code/O1-baseline-comparison/`

| KR | Deliverable | Status | Evidence |
|---|---|---|---|
| KR0 | Full-data baseline protocol approved/designed | Done | `6-code/O1-baseline-comparison/configs/O1_KR0_protocol.md` |
| KR1 | VRAM preflight and unified batch-size decision | Done | `results/vram_preflight.csv`, `results/vram_preflight_summary.md`; selected `batch_size=64` |
| KR2 | One-epoch smoke training for all five baselines | Done | `results/smoke_runs_bs64_e1/`, `results/smoke_training_time_estimate.md` |
| KR3 | Add progress bars to training script | Done | `scripts/train_baselines.py`, `code-modify.md` |
| KR4 | Design FNO-Large-bs64 fair-comparison experiment | Done | `scripts/train_fno_large_bs64.py`, O1 README |
| KR5 | Train five baselines for 50 epochs | Not started | Pending evening run |
| KR6 | Train FNO-Large-bs64 for 50 epochs | Blocked | Needs `tensorly-torch` dependency before run |
| KR7 | Evaluate baselines + FNO-Large-bs64 on same fixed test set | Not started | `scripts/evaluate_baselines.py` ready |
| KR8 | Compile plotting-ready O1 results | Planned | `scripts/collect_o1_results.py` ready |
| KR9 | Produce Table 4 + figures + §4.3 draft | Not started | Depends on KR5-KR8 |

Important current facts:

| Item | Value |
|---|---|
| Unified batch size | 64 |
| Sequential 50-epoch estimate for five baselines | about 43.3 h |
| Runtime bottleneck | BiLSTM, about 25.4 h for 50 epochs |
| FNO dependency issue | Current venv lacks `tensorly-torch` / `tltorch` |
| GitHub branch for server run | `HKUJasonJiang/Seismic-FNO.git`, branch `revise`, commit `02a023a` |
| Server run notes | `6-code/O1-baseline-comparison/configs/server_run_notes.md` on branch `revise` |

Immediate next actions:

1. On the server, clone/checkout `revise`.
2. Install/check `tensorly-torch` before FNO-Large-bs64 training.
3. Start formal O1 training when user approves evening run.
4. After training, run evaluation and `collect_o1_results.py`.

---

## O2: Model Data Efficiency Comparison

**Paper role**: `§5.1`  
**Active folder**: `6-code/O2-data-efficiency/`

| KR | Deliverable | Status | Evidence |
|---|---|---|---|
| KR0 | Data-efficiency protocol for FNO and baselines | Planned | `6-code/O2-data-efficiency/README.md` |
| KR1 | Existing Base efficiency results cleaned | Not started | Legacy CSV exists in archived code |
| KR2 | FNO-Large reduced-factor runs | Not started | Pending O1/FNO setup |
| KR3 | Baseline reduced-factor runs | Not started | Depends partly on O1 code |
| KR4 | Data-efficiency plot/table | Not started | Depends on KR1-KR3 |
| KR5 | Draft §5.1 text | Not started | Depends on KR4 |

Immediate next action: wait until O1 formal training is underway or complete; O2 can reuse O1 scripts after factor-count argument is added.

---

## O3: Generalization

**Paper role**: `§5.2`, `§5.3`  
**Active folder**: `6-code/O3-generalization/`

| KR | Deliverable | Status | Evidence |
|---|---|---|---|
| KR0 | OOD/PGA protocol | Not started | — |
| KR1 | OOD descriptors for K-NET + classic GMs | Not started | — |
| KR2 | Zero-shot evaluation using trained full/reduced models | Not started | Depends on O1/O2 checkpoints |
| KR3 | PGA extrapolation dataset split | Not started | — |
| KR4 | F-6 PGA extrapolation training/eval | Not started | — |
| KR5 | Optional FW-14 PGA extrapolation | Not started | — |
| KR6 | Draft §5.2-§5.3 text | Not started | — |

Immediate next action: after O1 checkpoints exist, start KR0/KR1 in parallel with evaluation.

---

## O4: Multi-Structure Validation

**Paper role**: `§5.4`  
**Active folder**: `6-code/O4-multi-structure/`

| KR | Deliverable | Status | Evidence |
|---|---|---|---|
| KR0 | Six-structure protocol and data-path workflow | Done | `6-code/O4-multi-structure/configs/O4_KR0_protocol.md`, `configs/o4_structures.json`, `configs/server_run_notes.md` |
| KR1 | Prepare training access and script for six HDF5 structures | Done | `scripts/inspect_o4_data.py`, `scripts/train_o4_fno_large.py`; local schema check passed; all six O4 HDF5 files uploaded to HF `JasonXF/SeFNO`; server handoff docs pushed to GitHub `revise` commit `e83a165` |
| KR2 | Smoke one epoch / limited batches for all six structures | Not started | Server run pending |
| KR3 | Train/evaluate all six structures for 50 epochs | Not started | Depends on KR2 |
| KR4 | Multi-structure convergence/summary table | Planned | `scripts/collect_o4_results.py` ready |
| KR5 | Draft §5.4 text | Not started | Depends on KR3-KR4 |

Immediate next action: server pulls `revise`, reads `6-code/O4-multi-structure/configs/server_takeover_o4.md`, downloads O4 data from HF, runs six-structure smoke, then full training.

Server code state:

- Repo: `https://github.com/HKUJasonJiang/Seismic-FNO.git`
- Branch: `revise`
- O4 workflow commit: `6a98315 Add O4 six-structure FNO training workflow`
- O4 handoff commit: `e83a165 Add O4 data upload and server handoff docs`
- HF dataset: `JasonXF/SeFNO`, path `MDOF/knet-250/Data/fno/`, six O4 HDF5 files verified uploaded on 2026-06-04.

---

## O5: FEMA P-58 Damage Assessment

**Paper role**: `§2.4`, `§6`  
**Active folder**: `6-code/O5-fema-p58/`

| KR | Deliverable | Status | Evidence |
|---|---|---|---|
| KR0 | Official FEMA/code parameter verification | In progress | `6-code/O5-fema-p58/resources/`; source pages summarized in `results/f6_verified/O5_FEMA_P58_workflow_teaching_report.md` |
| KR1 | Roof PFA extraction validation | Prototype only | Active F-6 workflow uses archived `workbuddy/legacy-results/fema_p58_results/pfa_F-6_roof_g.npy`; verified summary in `6-code/O5-fema-p58/results/f6_verified/f6_p58_verified_summary.md` |
| KR2 | Verified fragility pipeline rerun | Done for F-6 FEM roof-PFA; pending FNO/O4 extension | `scripts/extract_verified_p58_parameters.py`, `scripts/run_f6_p58_verified.py`, `results/verified_parameters/`, `results/f6_verified/` |
| KR3 | Hospital/F-6 PFA-sensitive NSC figure/table | Done for selected verified components | `6-code/O5-fema-p58/results/f6_verified/`, `6-code/O5-fema-p58/figures/f6_verified/` |
| KR4 | Data center FW-14 damage figure/table | Prototype only | Archived legacy FEMA outputs |
| KR5 | Damage-vs-height curve | Prototype only | Archived legacy FEMA outputs |
| KR6 | Code comparison | Prototype only | Archived legacy FEMA outputs |
| KR7 | Error propagation | Prototype only | Archived legacy FEMA outputs |
| KR8 | Draft §6 text | Not started | — |

Immediate next action: review the verified F-6 teaching report; next technical step is to add FNO-predicted PFA and record-level probability error once O1/O4 outputs exist.

---

## O6: Manuscript Assembly

**Paper role**: final manuscript  
**Active folder**: `6-code/O6-manuscript-assembly/`

| KR | Deliverable | Status | Evidence |
|---|---|---|---|
| KR0 | Final page/figure/table map | Not started | Depends on result availability |
| KR1 | Rewrite §1-§2 | Not started | — |
| KR2 | Update §3-§4 | Not started | Depends on O1 |
| KR3 | Integrate §5 | Not started | Depends on O2-O4 |
| KR4 | Integrate §6 | Not started | Depends on O5 |
| KR5 | Write §7-§8 | Not started | Depends on final results |
| KR6 | Compile ES manuscript | Not started | — |
| KR7 | Final package | Not started | — |

Immediate next action: skeleton only; avoid final claims before O1-O5 results exist.

---

## 4. Resume Checklist

When resuming this project, check in this order:

1. Read `.codex/progress-dashboard.md`.
2. If strategy changed, read/update `.codex/project-manage.md`.
3. For active experiment details, open the relevant `6-code/O*/README.md`.
4. For code changes, check the relevant `6-code/O*/code-modify.md`.
5. For O1 current state, check:
   - `6-code/O1-baseline-comparison/results/vram_preflight_summary.md`
   - `6-code/O1-baseline-comparison/results/smoke_training_time_estimate.md`
   - `6-code/O1-baseline-comparison/results/runs/`
   - `6-code/O1-baseline-comparison/results/compiled/`

---

## 5. Next Planned Work

Pending user approval for evening run:

1. O1 server side:
   `git clone -b revise https://github.com/HKUJasonJiang/Seismic-FNO.git`
2. Install/check `tensorly-torch` for FNO support.
3. Run O1 formal baseline/FNO-Large-bs64 training and evaluation.
4. Local/O4 side:
   run or transfer the six-structure O4 smoke commands from `6-code/O4-multi-structure/configs/server_run_notes.md`.
5. Local O5 side:
   keep the F-6-only P58 template current while official parameter verification and O4/FNO outputs are pending.
