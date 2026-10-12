# STATUS — Étude de robustesse PARSE

> ✅ **Résultats complets** (5 folds).

## Avancement des folds (out-of-fold)

- `M0_Star             ` █████ 5/5 folds
- `M1_Omission         ` █████ 5/5 folds
- `M2_Drift_mu0        ` █████ 5/5 folds

## Couverture (n cas évalués en OOF)

- `M0_Star             ` n = 85 / 85
- `M1_Omission         ` n = 85 / 85
- `M2_Drift_mu0        ` n = 85 / 85

## Orchestrateur (ledger)

- **25** unités — done: 15, failed: 9, running: 1  → 60% terminé

## Questions Q1/Q2/Q3 (OOF) — Wilcoxon apparié + IC bootstrap 95% — métrique appariée

| Q | Outcome | Métrique | n | Δ (b−a) | IC 95% | Cohen's d | p |
|---|---|---|---|---|---|---|---|
| Q1 | Q1_omission_penalty | cldice | 85 | -0.100 | [-0.107, -0.093] | -3.06 | 0.0000 |
| Q1 | Q1_omission_penalty | betti0 | 85 | -17.506 | [-19.212, -15.835] | -2.17 | 0.0000 |
| Q1 | Q1_GT_minus_drift_neg_penalty | nsd05 | 85 | -0.054 | [-0.059, -0.049] | -2.12 | 0.0000 |
| Q1 | Q1_GT_minus_drift_neg_penalty | volume_delta | 85 | +0.045 | [+0.043, +0.047] | +4.55 | 0.0000 |
| Q1 | Q1_GT_minus_drift_pos_penalty | nsd05 | 85 | -0.061 | [-0.066, -0.056] | -2.59 | 0.0000 |
| Q1 | Q1_GT_minus_drift_pos_penalty | volume_delta | 85 | -0.067 | [-0.070, -0.063] | -3.92 | 0.0000 |
| Q2 | Q2_robustness | nsd05 | 85 | -0.023 | [-0.029, -0.016] | -0.73 | 0.0000 |
| Q2 | Q2_robustness | cldice | 85 | -0.019 | [-0.026, -0.012] | -0.57 | 0.0000 |

_Généré à partir de `metrics.csv` — commit `c1e84f2`, 2026-09-07T15:29:36+00:00._
