# Public curated coefficient tables

Whitelist: formal labels **H1**, **H2**, **D1** only.

| formal_label | model_id | Primary / secondary specs |
|--------------|----------|---------------------------|
| H1 | `h1_engagement_indirect` | `main` (H1-M1, core sample); `h1_m2_composition` (adjustment); `h1_broader_sample` (appendix) |
| H2 | `h2_indirect_period` | `h2_m1_period` (main); `h2_m2_composition` (robustness) |
| D1 | `d1_indirect_peripheral` | `d1_m1_total_association` (main); `d1_m2_conditional_threshold` (robustness) |

Manuscript Q2 (Appendix D.1) is the broader peripheral sample: `table_h1_broader_sample.csv`, `table_q2_engagement_components_broader.csv`, and `table_q2_peripheral_thresholds_broader.csv`. The core-sample engagement equation (`main`, β 0.123) remains in `table_h1_m1_m2_comparison.csv` and is not Table D1.

Manuscript Q1 (Appendix D.2) keeps the composition-adjusted OR of 1.46 in `table_h2_m1_m2_comparison.csv`.

Engagement-component asymmetry (formal Wald tests that Indirect/Mixed coefficients for likes / comments / reposts are equal): `table_engagement_asymmetry_wald.csv`.

Expression validation and the H2 misclassification check (Appendix B):

- `table_expression_validation_period_rates.csv` — binary Direct vs Indirect/Mixed sensitivity and false-positive rate by period
- `table_misclassification_sensitivity_shares.csv` — Rogan–Gladen corrected shares (Table B6)
- `table_misclassification_sensitivity_or.csv` — imputed H2-M2 odds ratios

Corrected shares and odds ratios are sensitivity results. They do not replace the published 14.0% to 29.4% contrast or the odds ratio of 1.46.

Old-vs-new audit tables live under `outputs/reports/` (not public).
