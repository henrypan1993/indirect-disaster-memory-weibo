# Data availability

## Public materials

- Curated coefficient tables: `outputs/public/`
- Aggregate expression-validation rates and misclassification tables: `outputs/public/table_expression_validation_period_rates.csv`, `table_misclassification_sensitivity_shares.csv`, `table_misclassification_sensitivity_or.csv`
- Aggregate H2 counts (two rows): `data/public/h2_indirect_counts.csv`
- Selected model text dumps under `outputs/models/` (H1, H2-M1/M2, D1-M1/M2)
- Formal appendices A–D: `appendices/`

## Restricted materials (not redistributed)

- Post text and analysis text
- `mid`, `account_id`, and other account/post identifiers
- Full analysis-ready CSVs used for clustered standard errors
- Row-level validation records, including gold/LLM label pairs (`data/restricted/` is not published)
- Commercial API responses and credentials

## Rationale

Weibo content accessible on a platform is not automatically redistributable. This repository therefore supports **artifact verification** and documents the analysis code without publishing a post-level public microdata file. A future public cluster identifier or derived microdata file, if any, requires a separate re-identification review (deferred).
