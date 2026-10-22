# Reproducibility matrix

Levels:

- **Public** — a reader can check the published aggregate against the manuscript without restricted data. This is artifact verification, not a refit.
- **Restricted** — the result can be recomputed from privately held analysis files with the commands below. Those files are not in the public tree.
- **Not reproducible here** — this repository does not contain the inputs required to rebuild the result.

`uv run ruff check .` and `uv run pytest -q` (56 passed on 22 October 2026) exercise the code and synthetic fixtures. They do not refit the manuscript regressions.

| Result | Level | Source |
|--------|-------|--------|
| Corpus sequence 63,858 → 62,513 → 38,980 + 674 → 39,654 → 24,801 → 24,796 → 17,143 | Public as documented counts in Appendix A. The retrieval and backfill logs that produced them are not in this repository | Appendix A; `scripts/08_appendix_figures.py` |
| Q1 baseline OR 1.92 and adjusted OR 1.46 | Public artifact. Refit is restricted | `outputs/public/table_h2_m1_m2_comparison.csv`; `scripts/05_main_models.py` and `scripts/06_robustness_models.py` with `data/processed/model_data_final.csv` |
| Q1 shares 675/4,827 and 300/1,019 | Public aggregate | `data/public/h2_indirect_counts.csv` |
| Q2 total engagement β 0.132, N = 10,806 | Public artifact. Refit is restricted | `outputs/public/table_h1_broader_sample.csv`; spec `h1_broader_sample` |
| Q2 components 0.102 / 0.070 / 0.007, N = 10,806 | Public artifact. A restricted refit on 22 October 2026 matched the July 2026 archive and was then published. Core-sample component files were not replaced | `outputs/public/table_q2_engagement_components_broader.csv`; `uv run python scripts/06_robustness_models.py --q2-broader-only` |
| Appendix C engagement thresholds 0.126 / 0.132 / 0.130 | Public artifact, same recovery rule as the Q2 components | `outputs/public/table_q2_peripheral_thresholds_broader.csv` |
| Appendix C boundary thresholds | Public artifact from the current robustness summary | `outputs/public/table_peripheral_threshold_sensitivity.csv` (D1 rows) |
| Core-sample engagement β 0.123, N = 5,846 | Public artifact. This is not manuscript Table D1 | `outputs/public/table_h1_m1_m2_comparison.csv` |
| Stacked OLS F(2, 5475) = 7.15 and the three pairwise contrasts, core N = 5,846 | Public artifact. Synthetic tests cover the Wald helper. Refit is restricted. The test is not the broader-sample Table 3 contrast | `outputs/public/table_engagement_asymmetry_wald.csv`; `scripts/13_test_engagement_asymmetry.py` |
| Boundary OR 2.21, N = 17,067 | Public artifact. Refit is restricted | `outputs/public/table_d1_m1_m2_comparison.csv` |
| Appendix B.4 rates and confusion summaries | Public aggregates. Recomputing the rates from the 199 pairs is restricted | `outputs/public/table_expression_validation_period_rates.csv`; `scripts/14_misclassification_sensitivity.py` with `data/restricted/expression_validation_pairs.csv` |
| Appendix B.5 Rogan–Gladen shares and imputed ORs (seed 20261022; 80 fixed-rate imputations; 192 retained resamples) | Public aggregates. The imputation refit is restricted and was confirmed against the published tables | `table_misclassification_sensitivity_shares.csv`, `table_misclassification_sensitivity_or.csv` |
| Qualitative matrix and residual cases | Restricted. Outputs are gitignored and contain post-level fields | `scripts/12_export_qualitative_matrix.py`, `scripts/10_export_engagement_deviance_cases.py` |
| Raw retrieval, API backfill, and full LLM annotation | Not reproducible here | Upstream logs and prompts are outside this repository |

## Unresolved items

- Official Appendix D.1 prints the broader-sample hashtag coefficient as 0.035\*\*\*. The saved model text (`outputs/models/h1_engagement_indirect_h1_broader_sample.txt`) prints that p-value as 0.001. Under this repository’s rule, three stars require p < .001, so Appendix D.1 in the repository prints 0.035\*\*. The coefficient, standard error, and confidence interval are unchanged. The stars were not edited to match the official appendix.
- The manuscript’s stacked OLS paragraph uses the core sample (N = 5,846). Table 3 and Appendix D.1 component coefficients use the broader sample (N = 10,806). Both sets are published. They are not the same regression.
- An anonymized review manuscript is not in this repository. The title page outside the repository was not copied in. Public CSV headers were checked for post text, `mid`, and account identifiers.
- Main regression text dumps from `scripts/05_main_models.py` were not regenerated for this synchronization.
