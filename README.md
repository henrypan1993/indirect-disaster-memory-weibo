# Indirect Expression and Disaster-Memory Reactivation on Weibo

Analysis repository for the manuscript *Indirect Expression and Disaster-Memory Reactivation on Weibo: Zhengzhou Flood Discourse, 2021–2025*.

## Scope

This is an **outputs-first** companion repository. It documents the post-label statistical workflow, publishes curated coefficient tables for verification, and provides the analysis code. It does **not** redistribute raw Weibo posts, a complete historical archive, or a post-level public microdata file.

## Key findings (verification targets)

| Label | Result (main specification) |
|-------|-----------------------------|
| **H1** | Indirect/mixed expression associated with higher logged engagement among peripheral posts (β ≈ 0.132, ≈ +14.1%) |
| **H2** | T2 associated with higher odds of indirect/mixed expression among disaster-impact-related peripheral posts (H2-M1 OR ≈ 1.92; composition-adjusted H2-M2 OR ≈ 1.46) |
| **D1** | Peripheral accounts more likely to use indirect/mixed expression in the full Core sample (D1-M1 OR ≈ 2.21); the conditional threshold specification (D1-M2) is not significant |

Curated tables: [`outputs/public/`](outputs/public/). Full regression text dumps: selected files under `outputs/models/`.

## Reproducibility levels

1. **Public artifact verification** — Open curated CSVs in `outputs/public/` and Appendices A–D; confirm coefficients and Ns against the manuscript.
2. **Computational reproduction with restricted analytical data** — With a privately held analysis-ready table, run the workflow below via `uv`. Restricted inputs are not in this public tree.
3. **Non-public source-data reconstruction** — Rebuilding the corpus via commercial Weibo API, raw records, or full LLM annotation is **out of scope**.

## Repository structure

| Path | Contents |
|------|----------|
| `scripts/` | Pipeline `00`–`09`, `prepare_model_data.py`, and `common.py` |
| `tests/` | Unit tests and synthetic fixtures |
| `outputs/public/` | Curated coefficient tables (H1/H2/D1) |
| `outputs/models/` | Selected statsmodels text dumps |
| `appendices/` | Formal English Appendices A–D |
| `docs/` | Data availability, reproducibility, ethics, model-name mapping |
| `legacy/entropy_diagnostics/` | Excluded entropy materials (not confirmatory) |
| `data/README.md` | Why analytical microdata are absent |

## Manuscript mapping

| Formal label | Repository `model_id` |
|--------------|------------------------|
| H1 | `h1_engagement_indirect` |
| H2 | `h2_indirect_period` |
| D1 | `d1_indirect_peripheral` |

See [`docs/model_name_mapping.md`](docs/model_name_mapping.md) for `spec_id` variants and legacy aliases.

## Installation

Requires [uv](https://github.com/astral-sh/uv) and Python 3.11+ (pinned locally via `.python-version`).

```bash
uv sync
uv run ruff check .
uv run pytest -q
```

## Analysis workflow (tier 2 only)

With restricted input at `data/input/labels_core_cleaned.csv`:

```bash
uv run python scripts/00_check_input.py
uv run python scripts/01_build_analysis_ready.py
uv run python scripts/02_build_topic_entropy.py --from-entropy data/processed/analysis_ready_with_entropy.csv
# or recompute topics: uv run python scripts/02_build_topic_entropy.py
uv run python scripts/prepare_model_data.py
uv run python scripts/04_main_models.py
uv run python scripts/05_robustness_models.py --merge
```

Public curated tables: `uv run python scripts/_build_public_outputs.py` (local helper).

## Data availability

Restricted post-level data are **not** redistributed. See [`data/README.md`](data/README.md) and [`docs/data_availability.md`](docs/data_availability.md).

## Annotation

Gate/Core prompts and validation details are summarized in [Appendix B](appendices/Appendix_B.md). Upstream production prompts live outside this repository.

## License

Software code is licensed under MIT as stated in [`LICENSE-CODE`](LICENSE-CODE).

The MIT License applies only to software code. Manuscript text, appendices, and Weibo-derived materials remain under the authors’ copyright and are provided for scholarly verification.

## Citation

Cite the manuscript when using these materials. A `CITATION.cff` will be added after the manuscript metadata and author identifiers are finalized.
