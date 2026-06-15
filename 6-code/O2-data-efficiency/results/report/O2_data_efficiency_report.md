# O2 Data Efficiency Report

Final XG-Boost-1 / RTX PRO 6000 run synced on 2026-06-15.
The unified summary is o2_report_summary.csv.

## Scope

The final O2 package evaluates FNO-Large, Transformer, and BiLSTM under AC, GM, and joint reduced-data regimes on the fixed F-6 test set.

## Main Evidence Files

- o2_report_summary.csv: final machine-readable summary.
- o2_f6_data_efficiency_three_models.csv: original final summary table.
- figures/o2_f6_mse_by_percentage.png: MSE degradation by retained data percentage.
- figures/o2_f6_three_models_mse.png and o2_f6_three_models_r2.png: three-model comparison.
- figures/o2_f6_three_models_joint_mse.png: joint-reduction MSE comparison.

## Interpretation Note

The evidence is nuanced rather than a blanket FNO win.
Transformer remains strong under AC reduction, while FNO-Large is more stable than Transformer under severe GM and joint reduction. BiLSTM is competitive in joint reduction but has the highest memory and latency cost.
