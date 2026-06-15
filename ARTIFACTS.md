# SeFNO Paper Artifacts

Last updated: 2026-06-15

This repository branch records the code-side entry points for the paper artifact
sync. Large model checkpoints and result bundles are kept in the Hugging Face
dataset `JasonXF/SeFNO`, not in Git.

## Git Branches

- Current paper artifact branch: `sync/paper-artifacts-20260615`
- O2 final code/results branch: `sync/xgb1-o2-final` at `1fbde5f`
- O4 six-structure code/results branch: `sync/xgb1-o4` at `1911060`
- O1/O3 evidence branch: `sync/xgb2-o1-o3-o2partial` at `df66472`

## Hugging Face Dataset Paths

Canonical dataset repo:

```text
hf://datasets/JasonXF/SeFNO
```

Artifact runs:

```text
experiments/O1_baseline/2026-06-09_xgb2_full_bs64/
experiments/O2_data_efficiency/2026-06-09_xgb2_partial_EAC80/
experiments/O3_generalization/2026-06-09_xgb2_pga_sampling_gmood/
experiments/O4_multi_structure/2026-06-09_xgb1_fno_large_6struct/
experiments/O5_fema_p58/2026-06-09_local_F6_sample/
```

Pending upload after a write-scoped HF token is available:

```text
experiments/O2_data_efficiency/2026-06-15_xgb1_final/
experiments/O5_fema_p58/2026-06-15_local_multistructure/
```

## Checkpoint Coverage

- O1: baseline and FNO checkpoints are available in
  `experiments/O1_baseline/2026-06-09_xgb2_full_bs64/artifact/runs/`.
- O2: the latest XG-Boost-1 final report has no checkpoints under the remote
  O2 results directory. The older partial EAC80 run has FNO-Large and BiLSTM
  checkpoints in HF.
- O3: PGA and sampling OOD checkpoints are available in
  `experiments/O3_generalization/2026-06-09_xgb2_pga_sampling_gmood/artifact/`.
- O4: six-structure FNO-Large checkpoints and manifest are available in
  `experiments/O4_multi_structure/2026-06-09_xgb1_fno_large_6struct/`.
- O5: downstream FEMA P-58 results reference O4 F10/FW10 checkpoints through
  `6-code/O5-fema-p58/input/o4_artifacts/artifact_index.json`.

## Download Examples

Download only manifests and light report files:

```bash
hf download JasonXF/SeFNO \
  --repo-type dataset \
  --include "experiments/*/manifest.json" \
  --include "experiments/*/*/artifact/report/**" \
  --local-dir artifacts_hf
```

Download O4 checkpoints for multi-structure reproduction:

```bash
hf download JasonXF/SeFNO \
  --repo-type dataset \
  --include "experiments/O4_multi_structure/2026-06-09_xgb1_fno_large_6struct/runs/*/checkpoints/*_best.pt" \
  --local-dir artifacts_hf
```

## Code Entry Points

- O1 baseline comparison: `6-code/O1-baseline-comparison/scripts/`
- O2 data efficiency: `6-code/O2-data-efficiency/scripts/`
- O3 generalization: `6-code/O3-generalization/scripts/`
- O4 multi-structure: `6-code/O4-multi-structure/scripts/`
- O5 FEMA P-58 downstream assessment: `6-code/O5-fema-p58/scripts/`

Objective-specific README files under `6-code/O*/README.md` describe the paper
role and local result layout.
