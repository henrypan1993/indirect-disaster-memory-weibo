# Release changelog (manuscript synchronization, 22 October 2026)

The public repository was aligned with *Bounded Resonance in Urban Flood Risk Discourse* and the official Appendix A–D dated 22 October 2026. Fitted specifications were not renamed. The primary period estimate remains OR = 1.46. Main model dumps from script 05 were not regenerated.

## Documentation

- `README.md` — new title, Q1/Q2 verification table, corpus sequence, three reproducibility levels.
- `docs/model_name_mapping.md` — manuscript questions mapped onto existing `model_id` values.
- `docs/reproducibility.md`, `docs/data_availability.md`, `docs/ethics_and_privacy.md`, `data/README.md`, `data/public/README.md`.
- `docs/reproducibility_matrix.md` — this synchronization’s coverage and remaining gaps.
- `docs/updated_hypotheses_models_results.md` — banner stating that the internal H1/H2 note does not replace the revised manuscript.
- `appendices/Appendix_A.md` — Table A2 now includes 63,858 entries, the 674 retained 10 April statuses, 24,801 visible posts, and 24,796 Gate inputs.
- `appendices/Appendix_B.md` — B.5 subsections and Table B6 adjusted-OR column.
- `appendices/Appendix_C.md` — engagement column is the broader-sample threshold table.
- `appendices/Appendix_D.md` — D.1 broader engagement, D.2 period logit, D.3 boundary test.
- `appendices/README.md`, `outputs/public/README.md`, `outputs/public/MANIFEST.csv`.

## Code and public aggregates

- `scripts/08_appendix_figures.py` — sample-flow stages match Table A2.
- `scripts/14_misclassification_sensitivity.py` — reads `data/restricted/expression_validation_pairs.csv`. If that file is absent, the script exits without overwriting public tables. Row-level pairs were removed from `data/public/`.
- `scripts/06_robustness_models.py` — `--q2-broader-only` refits broader-sample component and threshold specs. On 22 October 2026 the refit matched the July 2026 archive, so these new public tables were written:
  - `outputs/public/table_q2_engagement_components_broader.csv`
  - `outputs/public/table_q2_peripheral_thresholds_broader.csv`
- Core-sample component files and `table_peripheral_threshold_sensitivity.csv` were not replaced.
- `.gitignore` ignores `data/restricted/`.

## Checks

- `uv run ruff check .` passed.
- `uv run pytest -q`: 56 passed. The suite includes synthetic misclassification and stacked-OLS tests and checks that public aggregates round to the manuscript figures. It does not reproduce the main regressions.
- `uv run python scripts/06_robustness_models.py --q2-broader-only` exited 0 after matching the archive.
- `uv run python scripts/14_misclassification_sensitivity.py` was rerun from the restricted validation file. The public aggregate checks still matched Appendix B (including the imputed median OR of 3.39).
