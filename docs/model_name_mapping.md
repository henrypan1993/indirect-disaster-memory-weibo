# Model name mapping

The 2026-10-22 manuscript uses research questions, not the repository’s H1/H2 strings. Internal `model_id` values are unchanged.

| Manuscript | Repository `model_id` | Spec used in the manuscript | Role |
|------------|----------------------|-----------------------------|------|
| Q1, Table 2, Appendix D.2 | `h2_indirect_period` | `h2_m1_period`; primary adjusted estimate `h2_m2_composition` (OR 1.46) | Indirect/Mixed ~ T2, disaster-impact × peripheral |
| Q2, Table 3, Appendix D.1 | `h1_engagement_indirect` | `h1_broader_sample`; components `engagement_*_broader` | Engagement ~ Indirect/Mixed, all peripheral posts |
| Stacked OLS paragraph | `h1_engagement_indirect` | script 13 on sample `h1` (core, N = 5,846) | Equality of likes / comments / reposts coefficients |
| Boundary test, Appendix D.3 | `d1_indirect_peripheral` | `d1_m1_total_association` | Indirect/Mixed ~ peripheral, full Core |

`main` (core engagement, β 0.123, N = 5,846) remains in the repository. It is not Appendix D.1.

## Spec variants (repository labels)

| Formal | `spec_id` | Role |
|--------|-----------|------|
| H1 | `main` | Core-sample engagement equation (not manuscript Table D1) |
| H1 | `h1_m2_composition` | Core-sample composition adjustment |
| H1 | `h1_broader_sample` | **Manuscript Q2 / Appendix D.1:** all peripheral posts |
| H1 | `engagement_likes` / `engagement_comments` / `engagement_reposts` | Component DVs on the core sample |
| H1 | `engagement_likes_broader` / `engagement_comments_broader` / `engagement_reposts_broader` | **Manuscript Table 3 / Appendix D.1 components** |
| H2 | `h2_m1_period` | Q1 baseline (OR 1.92) |
| H2 | `h2_m2_composition` | **Q1 primary adjusted estimate (OR 1.46)** |
| D1 | `d1_m1_total_association` | Primary boundary model |
| D1 | `d1_m2_conditional_threshold` | Conditional: adds verified + log_followers |

## Expectation / residual case selection (script 10; not confirmatory)

The expectation equation uses the core-sample engagement formula. It is for residual case selection, not manuscript Table D1.

| `model_id` | `spec_id` | Role |
|------------|-----------|------|
| `expectation_engagement_expression` | `expectation_main` | Same formula as H1-M1; used for row-level residuals |
| `expectation_engagement_expression` | `expectation_with_relative_day` | + `relative_window_day`; rank-stability check only |

Outputs under `outputs/cases/` (restricted / gitignored).

## Sample flags

| Flag | Definition |
|------|------------|
| `sample_h1_eligible` / `sample_h2_eligible` | Core: `peripheral` ∩ `disaster_impact_related` ∩ non-missing expression |
| `sample_h1_broader_eligible` | Broader: `peripheral` ∩ non-missing expression (any narrative) |
| `disaster_impact_related` | Alias of former `narrative_trauma_clean` (`Trauma-Help-Loss` or `Memory-Reactivation`) |

## Semantic analysis IDs (qualitative sampling / joint display)

From the qualitative matrix (script 12) onward, reports use **semantic IDs**.
Legacy repository `model_id` values (`h1_*`, `h2_*`) are **not renamed**; they remain
compatibility identifiers for existing model dumps and public tables.

| Semantic ID | Manuscript question | Legacy `model_id` | Meaning |
|-------------|---------------------|-------------------|---------|
| `period_to_indirect` | **Q1** | `h2_indirect_period` | Risk context (T2) → Indirect/Mixed |
| `indirect_to_engagement` | **Q2** | `h1_engagement_indirect` | Indirect/Mixed → engagement |
| `composition_exploration` | (exploratory, not confirmatory) | — | Script 11: peripheral Indirect/Mixed composition |

Q1 baseline OR ≈ 1.92; composition-adjusted OR = 1.46 (`h2_m1_period` / `h2_m2_composition`). Appendix B and `scripts/14_misclassification_sensitivity.py` check that contrast for misclassification. Corrected odds ratios there do not replace 1.46.
Q2 component pattern on the broader sample: likes 0.102 and comments 0.070; reposts 0.007 (`engagement_*_broader`).
The stacked OLS Wald test is on the core sample, not on those broader-sample coefficients:
`scripts/13_test_engagement_asymmetry.py` →
`outputs/public/table_engagement_asymmetry_wald.csv`.

Public CSV column `formal_label` still stores the repository strings H1, H2, and D1.
Read those strings through the manuscript table at the top of this file.

### Qualitative strata (script 12)

- **A1** (`A1_contextual_transformation`): period-matched Indirect (broader sample),
  including Ordinary-Weather-Info in both years. Eligibility ignores engagement.
- **A2** (`A2_indirect_form_contrast` / `A2_direct_form_contrast`): Direct vs Indirect
  inside the core disaster-impact universe only (`Trauma-Help-Loss` ∪
  `Memory-Reactivation`); each period×form cell is 6 Trauma + 2 Memory. A2 Indirect
  is a subset of A1 selected core Indirect.
- **B** (`B_bounded_resonance`): Indirect with `r_L>0`, `r_C>0`, `r_R≤0`, ranked by
  period-internal `S_B = z(r_L)+z(r_C)-z(r_R)`. `repost_share` is descriptive only.
- **C** (`C_portability_deviant`): period-specific P90 of `reposts_resid` and
  `reposts_count >= 2`.
- **Contrasts** (`low_engagement_contrast`): period-median predicted engagement and
  period P10 residual. Never labeled `failed_resonance`.
- **Other-Unclear** (`sensitivity_unclear`) is not a principal matched stratum
  (residual category; 2–3 sensitivity cases).
- One row per `mid`; `selection_role` is multi-value. Manual exclusions are restricted
  to: `duplicate_redundant_discourse`, `insufficient_analyzable_text`,
  `account_duplication`, `coding_ambiguity`, `maximum_variation_balancing`.
  Sampling is not allowed to drop cases for lacking the four theoretical features.

## Former `model_id` aliases

| Current `model_id` | Former `model_id` |
|--------------------|-------------------|
| `h2_indirect_period` | `h2b_indirect_reactivation` |
| `d1_indirect_peripheral` | `e1b_indirect_peripheral` |
