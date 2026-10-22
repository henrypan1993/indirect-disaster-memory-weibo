# Data directory

## What is not here

This public repository does **not** include:

- `data/input/labels_core_cleaned.csv` (contains post text, `mid`, `account_id`, and related fields);
- `data/processed/` analysis-ready tables (inherit text and identifiers);
- `data/reports/` machine-local audit JSON (may contain absolute paths);
- `data/restricted/expression_validation_pairs.csv` (row-level gold and LLM expression labels).

Those files remain on restricted research systems. They are required for **tier-2 computational reproduction** (re-fitting models with account-clustered standard errors) but are not redistributed.

## What is published instead

Curated aggregate coefficient tables under [`outputs/public/`](../outputs/public/), selected full regression text dumps for the engagement, period, and boundary models, and the two-row H2 count file in [`data/public/`](public/README.md). Row-level validation pairs are not published.

## Access

If an institutional data-access pathway is established, it will be documented here. Until then, treat post-level Weibo-derived materials as **not available** for public download.
