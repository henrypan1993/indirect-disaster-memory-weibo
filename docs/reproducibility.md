# Reproducibility

## Three levels

1. **Public artifact verification** — Compare manuscript coefficients to `outputs/public/*.csv` and Appendices A–D. No restricted data required.
2. **Computational reproduction with restricted analytical data** — Place the restricted cleaned label table at `data/input/labels_core_cleaned.csv`, then run the workflow below with `uv run`.
3. **Non-public source-data reconstruction** — Historical search, API backfill, and full LLM annotation are not reproducible from this repository alone.

## Formal modeling workflow

```text
labels_core_cleaned
→ 01_build_analysis_ready
→ 02 topic assignment → analysis_ready_with_topics.csv
→ prepare_model_data.py → model_data_final.csv
→ 04_main_models (H1, H2-M1, D1-M1)
→ 05_robustness_models (H1 components, H2-M2, D1-M2, peripheral thresholds)
```

Do not use `analysis_ready_with_entropy.csv` as the formal modeling entry point. Entropy columns belong to the excluded legacy path (`legacy/entropy_diagnostics/`).

## Seeds

- Topic K-Means: `random_state=42` in `scripts/common.py`
- Figure-4 bootstrap (if used): `--seed 12345` in `scripts/08_fig4_engagement_return.py`

## Curated public tables

Public CSVs are whitelist-filtered to:

`h1_engagement_indirect`, `h2_indirect_period`, `d1_indirect_peripheral`

mapped to formal labels H1, H2, D1. Spec variants (`h2_m1_period`, `h2_m2_composition`, `d1_m1_total_association`, `d1_m2_conditional_threshold`) are labeled in the public comparison tables.

Internal old-vs-new audits: `outputs/reports/` (not published as confirmatory results).

## Figures

No figure binaries are shipped in the current public tree. Manuscript figures must be regenerated and privacy-reviewed before any future binary release.
