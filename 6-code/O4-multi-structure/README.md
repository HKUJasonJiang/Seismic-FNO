# O4 Multi-Structure Validation

Paper role: `§5.4`; upstream result package for `§6`.

Purpose: show that FNO-Large is not limited to the original F-6 structure.

Current rule:

- O4 is a six-structure validation package, not only F-6/FW-14.
- O5 FEMA/FRS details should wait until O4 training/evaluation is underway or complete.
- Main protocol: `configs/O4_KR0_protocol.md`.
- Structure map: `configs/o4_structures.json`.
- Server commands: `configs/server_run_notes.md`.

Target structures:

| ID | Type | Stories |
|---|---|---:|
| F-2 | Frame | 2 |
| F-6 | Frame | 6 |
| F-10 | Frame | 10 |
| FW-10 | Frame-shear wall | 10 |
| FW-14 | Frame-shear wall | 14 |
| FW-17 | Frame-shear wall | 17 |

Expected paper deliverables:

- Table 6: multi-structure metrics.
- Convergence/summary figure.
- Draft text for `§5.4`.

Active scripts:

- `scripts/inspect_o4_data.py`
- `scripts/train_o4_fno_large.py`
