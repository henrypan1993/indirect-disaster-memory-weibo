# Reproducibility

## Three levels

1. **Public artifact verification** — Compare manuscript coefficients to `outputs/public/*.csv` and Appendices A–D. No restricted data required.
2. **Computational reproduction with restricted analytical data** — Place the restricted cleaned label table at `data/input/labels_core_cleaned.csv`, then run the workflow below with `uv run`.
3. **Non-public source-data reconstruction** — Historical search, API backfill, and full LLM annotation are not reproducible from this repository alone.

Verified corpus sequence (Appendix A): 63,858 retrieval entries → 62,513 unique IDs → 38,980 statuses from the 11 April 2026 backfill plus 674 statuses retained from 39 batch files written on 10 April → 39,654 consolidated posts → 24,801 visible posts → 24,796 Gate inputs → 17,143 Core posts. The 674 records are retained statuses, not a field-merging artifact.

## Formal modeling workflow

```text
labels_core_cleaned
→ 01_build_analysis_ready
→ 02_build_topic_assignment → analysis_ready_with_topics.csv
→ 03_prepare_model_data → model_data_final.csv
→ 05_main_models (H1-M1, H2-M1, D1-M1; H1/H2 share disaster-impact × peripheral core)
→ 06_robustness_models (H1 components, H1-M2, H2-M2, broader H1, D1-M2, peripheral thresholds)
→ 08_appendix_figures (manuscript Figure 3 / A1–A2)
→ 10_export_engagement_deviance_cases (optional; residual deviance cases)
→ 11_explore_indirect_composition (optional; exploratory composition of peripheral Indirect/Mixed)
→ 12_export_qualitative_matrix (optional; A1/A2/B/C qualitative sampling matrix)
→ 13_test_engagement_asymmetry (optional; formal Wald tests of likes/comments/reposts coef equality)
→ 14_misclassification_sensitivity (Appendix B; share correction is public, OR refit is tier 2)
```

Step 10 fits the legacy expectation OLS on the core sample, screens viral/review outliers, and exports top/bottom residual Indirect-Mixed cases under `outputs/cases/` (restricted; gitignored). A `+ relative_window_day` spec is used only to check residual-rank stability.

Step 11 is an **exploratory correlational** follow-up (`composition_exploration`, not confirmatory): on Peripheral × Indirect/Mixed (no `disaster_impact_related` filter), it compares 2021 vs 2025 narrative/emotion composition (χ² + Cramér's V) and estimates engagement associations (M2a reactivation, M2b narrative FE, M2c reactivation×T2). Outputs under `outputs/tables/` and `outputs/models/indirect_composition_*`.

Step 12 builds the qualitative matrix used for mixed-methods integration (`scripts/12_export_qualitative_matrix.py`). Semantic IDs: `period_to_indirect` (paper H1) and `indirect_to_engagement` (paper H2); legacy `h1_`/`h2_` model ids are unchanged. A1/A2 ignore engagement (A2 Indirect is drawn from A1 core Trauma/Memory); B ranks period-internal `S_B` among sign-conditioned residuals; C uses period P90 of `reposts_resid` with `reposts_count >= 2`; contrasts use period-median predicted engagement and P10 residual. One row per `mid` with multi-value `selection_role`; manual exclusions are a closed set. Outputs: `outputs/cases/qualitative_matrix.xlsx` (sheets `unique_cases` / `a1` / `a2` / `b` / `c` / `contrasts` / `selection_log`) and `qualitative_selection_log.csv` (gitignore). Other-Unclear is sensitivity-only, not a matched A1 stratum.

Step 13 formally tests engagement-component asymmetry on the disaster-impact × peripheral core sample (`scripts/13_test_engagement_asymmetry.py`, N = 5,846). It stacks likes/comments/reposts equations (outcome × predictors; reference = reposts) with account-clustered SEs and reports Wald tests of Indirect/Mixed coefficient equality (joint and pairwise). Separate-equation point estimates are cross-checked against the stacked model. This test does not use the broader-sample coefficients in manuscript Table 3. Outputs: `outputs/tables/table_engagement_asymmetry_wald.csv`, `table_engagement_asymmetry_coefs.csv`, and `outputs/public/table_engagement_asymmetry_wald.csv`.

Step 14 checks whether expression misclassification could have produced the Q1 period contrast (`scripts/14_misclassification_sensitivity.py`). When `data/restricted/expression_validation_pairs.csv` is present, the script computes wave-specific sensitivity and false-positive rates from those pairs. It does not insert manuscript rates in place of that calculation. Rogan–Gladen share correction also uses `data/public/h2_indirect_counts.csv`. Re-fitting `h2_m2_composition` on imputed labels requires `data/processed/model_data_final.csv`. Imputation uses seed `20261022`, 80 draws at fixed rates, and 200 within-wave resamples of the validation pairs (192 retained in the published run). If the restricted validation file is absent, the script exits without overwriting public tables. Outputs already published: `outputs/public/table_expression_validation_period_rates.csv`, `table_misclassification_sensitivity_shares.csv`, and `table_misclassification_sensitivity_or.csv`. Corrected shares and odds ratios do not replace 14.0% to 29.4% or OR 1.46.

`scripts/06_robustness_models.py --q2-broader-only` refits manuscript Q2 component and peripheral-threshold equations on the broader sample under new spec IDs. It writes `outputs/public/table_q2_engagement_components_broader.csv` and `table_q2_peripheral_thresholds_broader.csv` only when the refit matches the July 2026 archive. It does not replace core-sample component files.

Topic assignment supplies `topic_id` for fixed effects in the regression models.

## Seeds

- Topic K-Means: `random_state=42` in `scripts/common.py`
- Qualitative matrix (script 12): `--seed` default `12`
- Misclassification imputation (script 14): `20261022`

## Curated public tables

Public CSVs are whitelist-filtered to:

`h1_engagement_indirect`, `h2_indirect_period`, `d1_indirect_peripheral`

mapped to formal labels H1, H2, D1. Spec variants (`main` / H1-M1, `h1_m2_composition`, `h1_broader_sample`, `h2_m1_period`, `h2_m2_composition`, `d1_m1_total_association`, `d1_m2_conditional_threshold`) are labeled in the public comparison tables.

Internal old-vs-new audits: `outputs/reports/` (not published as confirmatory results).

## Figures

No figure binaries are shipped in the current public tree. Manuscript figures must be regenerated and privacy-reviewed before any future binary release.
