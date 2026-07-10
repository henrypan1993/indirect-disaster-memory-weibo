# Indirect Expression and Disaster-Memory Reactivation on Weibo

Analysis repository for the manuscript *Indirect Expression and Disaster-Memory Reactivation on Weibo: Zhengzhou Flood Discourse, 2021–2025*.

## Scope

This is an **outputs-first** companion repository. It documents the post-label statistical workflow, publishes curated coefficient tables for verification, and provides the analysis code. It does **not** redistribute raw Weibo posts, a complete historical archive, or a post-level public microdata file.

## Key findings (verification targets)

| Label | Result (main specification) |
|-------|-----------------------------|
| **H1** | Indirect/mixed expression associated with higher logged engagement among peripheral posts (β ≈ 0.132, ≈ +14.1%) |
| **H2** | T2 associated with higher odds of indirect/mixed expression among peripheral trauma-related posts (OR ≈ 2.41) |
| **D1** | Slightly lower topic entropy in T2 (β ≈ −0.010) |
| **D2** | Entropy does not explain the H1 engagement association (entropy β ≈ −0.090, ns) |
| **D3** | Peripheral account position does not significantly predict indirect/mixed expression (β ≈ 1.246, ns) |

Curated tables: [`outputs/public/`](outputs/public/). Full regression text dumps for the five models above: selected files under `outputs/models/`.

## Reproducibility levels

1. **Public artifact verification** — Open curated CSVs in `outputs/public/` and Appendices A–E; confirm coefficients and Ns against the manuscript. This is **verification**, not full computational reproduction.
2. **Computational reproduction with restricted analytical data** — With a privately held analysis-ready table (including fields required for account-clustered SEs), run `scripts/00`–`06` via `uv`. Restricted inputs are not in this public tree.
3. **Non-public source-data reconstruction** — Rebuilding the corpus via commercial Weibo API, raw records, or full LLM annotation is **out of scope** and is not claimed as reproducible from public materials.

## Repository structure

| Path | Contents |
|------|----------|
| `scripts/` | Pipeline `00`–`09` and `common.py` |
| `tests/` | Unit tests and synthetic fixtures |
| `outputs/public/` | Curated coefficient tables (whitelist: H1/H2/D1/D2/D3) |
| `outputs/models/` | Selected full statsmodels text dumps (five models only) |
| `appendices/` | Formal English Appendices A–E |
| `docs/` | Data availability, reproducibility, ethics, model-name mapping |
| `data/README.md` | Why analytical microdata are absent |

## Manuscript mapping

| Formal label | Repository `model_id` |
|--------------|------------------------|
| H1 | `h1_engagement_indirect` |
| H2 | `h2b_indirect_reactivation` |
| D1 | `h2a_entropy_reactivation` |
| D2 | `e2_entropy_increment` |
| D3 | `e1b_indirect_peripheral` |
| Robustness | Curated rows in `outputs/public/table_robustness_focal.csv` and related files |

See [`docs/model_name_mapping.md`](docs/model_name_mapping.md) for repository `model_id` strings and excluded legacy outputs.

## Installation

Requires [uv](https://github.com/astral-sh/uv) and Python 3.11+ (pinned locally via `.python-version`).

```bash
uv sync
uv run ruff check .
uv run pytest -q
```

## Analysis workflow (tier 2 only)

With restricted input present locally at `data/input/labels_core_cleaned.csv`:

```bash
uv run python scripts/00_check_input.py
uv run python scripts/01_build_analysis_ready.py
uv run python scripts/02_build_topic_entropy.py
uv run python scripts/04_main_models.py
uv run python scripts/05_robustness_models.py --merge
uv run python scripts/06_build_regression_tables.py
```

Public curated tables are built from those outputs using a whitelist of the five formal models (see `docs/reproducibility.md`).

## Data availability

Restricted post-level data (including raw text and account identifiers) are **not** redistributed. See [`data/README.md`](data/README.md) and [`docs/data_availability.md`](docs/data_availability.md).

## Annotation

Gate/Core prompts and validation details are summarized in [Appendix B](appendices/Appendix_B.md). Upstream production prompts live outside this repository; any future public prompt copies will be labeled **verbatim** or **privacy-redacted**.

## License

Software code is licensed under MIT as stated in [`LICENSE-CODE`](LICENSE-CODE).

**`LICENSE-CODE` applies only to software code.** It does **not** apply to manuscript text, appendices, article-derived documentation, Weibo-derived materials, curated numerical release tables beyond ordinary scholarly quotation, or third-party content.

## Citation

Cite the manuscript when using these materials. A `CITATION.cff` may be added when the public repository URL and author metadata are finalized.
