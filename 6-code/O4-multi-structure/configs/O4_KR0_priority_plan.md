# O4 KR0 Priority Plan: Superseded Note

**Version**: v0.1  
**Date**: 2026-06-04  
**Status**: Superseded by `O4_KR0_protocol.md`  
**Reason**: O5 FEMA/FRS figures need credible FNO outputs for the selected case structures.

---

## Supersession

This memo was an intermediate correction that over-emphasized F-6 and FW-14. The advisor clarified that O4 itself consists of six multi-structure experiments, and none should be treated as already done.

Use the full protocol instead:

```text
6-code/O4-multi-structure/configs/O4_KR0_protocol.md
```

## 1. Critical-Path Correction

O5 FEMA P-58 methodology can be reviewed in parallel, but final O5 scenario figures should not be produced before the case-structure FNO outputs exist.

For the revised paper, the practical dependency is:

```text
FNO-Large works fairly against baselines
  -> train/evaluate FNO-Large on F-6 and FW-14
    -> extract FEM/FNO roof or floor acceleration time histories
      -> compute PFA and FRS
        -> FEMA damage probabilities and FRS design interpretation
```

Therefore, F-6 and FW-14 are priority O4 runs because they feed both:

- `§5.4` multi-structure validation,
- `§6` hospital/data-center FEMA/FRS engineering examples.

---

## 2. Priority Structures

| Structure | Paper Role | Why Priority |
|---|---|---|
| F-6 | Main continuity case | Links old manuscript, baseline comparison, existing figures, and hospital-style example |
| FW-14 | Engineering case structure | Supports data-center / taller frame-wall example and extends beyond the original F-6 |

Remaining structures such as F-2, F-10, FW-10, and FW-17 are still useful for the broader multi-structure table, but they do not need to block the first FEMA/FRS scenario pipeline.

---

## 3. Required Outputs Before O5 Final Figures

For each priority structure:

1. Same or documented FNO-Large architecture/hyperparameters.
2. Fixed train/validation/test split with recorded seed and factor count.
3. Test-set FEM and FNO acceleration time histories.
4. PFA table:
   - sample ID,
   - floor or roof level,
   - FEM PFA,
   - FNO PFA,
   - absolute and relative error.
5. FRS table/array:
   - oscillator frequency grid,
   - damping ratio,
   - FEM FRS,
   - FNO FRS,
   - median and 84th-percentile curves across selected records.

---

## 4. Figure Implications

Existing or partially existing:

- Time-history figure exists.
- FRS figure exists, but needs median and 84th-percentile summary.

Missing and dependent on O4/O5:

- Damage probability comparison figure.
- FRS frequency-demand heatmap or median/84th-percentile spectra.
- PFA/FRS error-propagation figure.

---

## 5. Immediate To-Do

1. Confirm where F-6 and FW-14 HDF5 files live.
2. Confirm whether F-6 should be retrained with the same FNO-Large-bs64 protocol as O1 or whether the existing F-6 checkpoint is acceptable for non-baseline engineering figures.
3. Prepare server-ready commands for F-6 and FW-14 FNO-Large training/evaluation.
4. Add post-processing hooks for PFA and FRS median/84th-percentile outputs.
