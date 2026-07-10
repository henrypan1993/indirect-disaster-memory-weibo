# Reproducibility

## Three levels

1. **Public artifact verification** — Compare manuscript coefficients to `outputs/public/*.csv` and Appendices A–E. No restricted data required. This is not full reproduction.
2. **Computational reproduction with restricted analytical data** — Place the restricted cleaned label table at `data/input/labels_core_cleaned.csv`, then run `scripts/00`–`06` with `uv run`. Account-clustered SEs require account identifiers in that private file.
3. **Non-public source-data reconstruction** — Historical search, `statuses/show_batch`, and full LLM annotation require commercial API access and upstream projects; not reproducible from this repository alone.

## Seeds

- Topic K-Means: `random_state=42` in `scripts/common.py`
- Figure-4 bootstrap (if used): `--seed 12345` in `scripts/08_fig4_engagement_return.py`

## Curated public tables

Public CSVs are produced by whitelist filtering to:

`h1_engagement_indirect`, `h2b_indirect_reactivation`, `h2a_entropy_reactivation`, `e2_entropy_increment`, `e1b_indirect_peripheral`

mapped to formal labels H1, H2, D1, D2, D3. Original analysis outputs under `outputs/tables/` and `outputs/models/` that contain other models are not published.

## Figures

No figure binaries are shipped in the current public tree (`outputs/figures/` empty at packaging). Manuscript figures must be regenerated and privacy-reviewed before any future binary release. Any manual post-processing will be documented here.
