# Bounded Resonance in Urban Flood Risk Discourse

Analysis repository for the manuscript *Bounded Resonance in Urban Flood Risk Discourse: A Mixed-Methods Study with Implications for Social Impact Assessment*.

## Scope

This is an **outputs-first** companion repository. It documents the post-label statistical workflow, publishes curated coefficient tables for verification, and provides the analysis code. It does **not** redistribute raw Weibo posts, a complete historical archive, post-level microdata, or row-level validation records.

Corpus construction, verified from the retrieval and backfill logs and stated in Appendix A:

63,858 retrieval entries → 62,513 unique post IDs → 38,980 status records from the 11 April 2026 backfill, plus 674 non-overlapping status records retained from 39 batch files written on 10 April → 39,654 consolidated posts → 24,801 visible posts → 24,796 Gate inputs → 17,143 Core posts.

The 674 records are retained statuses from the preceding backfill. They are not a field-merging artifact.

## Key findings (verification targets)

| Manuscript result | Estimate | Where to check |
|-------------------|----------|----------------|
| **Q1.** Indirect/Mixed share, disaster-impact posts from peripheral accounts | 14.0% (675/4,827) in 2021 to 29.4% (300/1,019) in 2025. Baseline OR 1.92 (N = 5,846). Composition-adjusted OR **1.46**, 95% CI [1.17, 1.84] (N = 5,804) | `outputs/public/table_h2_m1_m2_comparison.csv` |
| **Q2.** Indirect/Mixed and logged engagement, broader peripheral sample | β = 0.132 (SE 0.029), N = 10,806, about +14.1% | `outputs/public/table_h1_broader_sample.csv` |
| **Q2 components** on that broader sample | Likes 0.102; comments 0.070; reposts 0.007 (N = 10,806) | `outputs/public/table_q2_engagement_components_broader.csv` |
| **Engagement asymmetry** (stacked OLS, core sample, N = 5,846) | Joint F(2, 5475) = 7.15; likes vs reposts 14.07; comments vs reposts 6.23; likes vs comments 3.46 | `uv run python scripts/13_test_engagement_asymmetry.py` → `outputs/public/table_engagement_asymmetry_wald.csv` |
| **Boundary test** | Peripheral OR 2.21 (N = 17,067). The conditional specification is not significant | `outputs/public/table_d1_m1_m2_comparison.csv` |

The composition-adjusted OR of 1.46 is the primary period estimate. Appendix B.5 and `outputs/public/table_misclassification_sensitivity_*.csv` are conditional sensitivity results. They do not replace 14.0% to 29.4% or OR 1.46.

The stacked test uses the disaster-impact × peripheral core sample. It does not test the broader-sample coefficients in Q2 / Table 3.

Internal model IDs are unchanged: Q1 is `h2_indirect_period`, Q2 is `h1_engagement_indirect`, and the boundary test is `d1_indirect_peripheral`. See [`docs/model_name_mapping.md`](docs/model_name_mapping.md).

## Reproducibility levels

1. **Public artifact verification** — Open curated CSVs in `outputs/public/` and Appendices A–D. No restricted data are required. This checks published aggregates. It does not refit the models.
2. **Computational reproduction with restricted analytical data** — With the privately held analysis-ready table and, for Appendix B, the restricted validation file, run the workflow below via `uv`.
3. **Non-public source-data reconstruction** — Rebuilding the corpus via the commercial Weibo API, raw records, or full LLM annotation is **out of scope**.

Which results fall in each level: [`docs/reproducibility_matrix.md`](docs/reproducibility_matrix.md).

## Repository structure

| Path | Contents |
|------|----------|
| `scripts/` | Numbered pipeline `00`–`14`, plus `common.py` |
| `tests/` | Unit tests and synthetic fixtures |
| `outputs/public/` | Curated coefficient, validation-rate, and sensitivity tables |
| `data/public/` | Aggregate H2 counts only (no post text, no validation pairs) |
| `outputs/models/` | Selected statsmodels text dumps |
| `appendices/` | Formal English Appendices A–D |
| `docs/` | Data availability, reproducibility, ethics, model-name mapping, release notes |
| `data/README.md` | Why analytical microdata are absent |

## Installation

Requires [uv](https://github.com/astral-sh/uv) and Python 3.11+.

```bash
uv sync
uv run ruff check .
uv run pytest -q
```

`ruff` and `pytest` check code and synthetic fixtures. They do not reproduce the manuscript regressions.

## Analysis workflow (tier 2)

With restricted input at `data/input/labels_core_cleaned.csv`:

```bash
uv run python scripts/00_check_input.py
uv run python scripts/01_build_analysis_ready.py
uv run python scripts/02_build_topic_assignment.py
uv run python scripts/03_prepare_model_data.py
uv run python scripts/05_main_models.py
uv run python scripts/06_robustness_models.py --merge
```

Optional descriptive tables: `uv run python scripts/04_descriptive_tables.py`.

Manuscript figures: `uv run python scripts/08_appendix_figures.py`.

Regression table exports: `uv run python scripts/07_build_regression_tables.py`.

Residual deviance cases (legacy qualitative export; writes gitignored `outputs/cases/`):

```bash
uv run python scripts/10_export_engagement_deviance_cases.py
```

Exploratory composition of peripheral Indirect/Mixed (not confirmatory):

```bash
uv run python scripts/11_explore_indirect_composition.py
```

Qualitative matrix (semantic IDs; gitignored `outputs/cases/`):

```bash
uv run python scripts/12_export_qualitative_matrix.py
```

Stacked OLS Wald tests:

```bash
uv run python scripts/13_test_engagement_asymmetry.py
```

Misclassification sensitivity (Appendix B.5). Rates are computed from the restricted validation file when it is present at `data/restricted/expression_validation_pairs.csv`. The script does not substitute manuscript constants for that calculation. Share correction also uses `data/public/h2_indirect_counts.csv`. Refitting the composition-adjusted logit requires `data/processed/model_data_final.csv`.

```bash
uv run python scripts/14_misclassification_sensitivity.py
```

Public curated tables: `uv run python scripts/_build_public_outputs.py` (local helper; not part of the public script list).

## Data availability

Restricted post-level data and row-level validation records are **not** redistributed. See [`data/README.md`](data/README.md) and [`docs/data_availability.md`](docs/data_availability.md).

## Annotation

Gate/Core prompts and validation details are summarized in [Appendix B](appendices/Appendix_B.md). Upstream production prompts live outside this repository.

## License

Software code is licensed under MIT as stated in [`LICENSE-CODE`](LICENSE-CODE).

The MIT License applies only to software code. Manuscript text, appendices, and Weibo-derived materials remain under the authors’ copyright and are provided for scholarly verification.

## Citation

Cite the manuscript when using these materials. A `CITATION.cff` will be added after the manuscript metadata and author identifiers are finalized. This repository does not include an anonymized review manuscript or a title page.
