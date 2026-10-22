# Appendices (English)

Formal English appendices for the manuscript.

| File | Role |
|------|------|
| [Appendix_A.md](Appendix_A.md) | Corpus construction and sample screening (Tables A1–A2) |
| [Appendix_B.md](Appendix_B.md) | Expression coding, period-stratified validation, and misclassification sensitivity (Tables B1–B6) |
| [Appendix_C.md](Appendix_C.md) | Robustness to alternative peripheral thresholds (Table C1) |
| [Appendix_D.md](Appendix_D.md) | Complete regression results (D.1 broader engagement, D.2 period logit, D.3 boundary test) |

## Formal labels (repository mapping)

| Label | Meaning |
|-------|---------|
| Q2 / repository H1 | Engagement ~ Indirect/Mixed. Manuscript table: broader peripheral sample (`h1_broader_sample`) |
| Q1 / repository H2 | Indirect/Mixed ~ T2. Primary adjusted OR = 1.46 (`h2_m2_composition`) |
| Boundary test | Indirect/Mixed ~ peripheral (full Core; primary / conditional) |

Repository `model_id` strings: [`docs/model_name_mapping.md`](../docs/model_name_mapping.md).

## Statistical conventions

Clustered robust standard errors by account. \* *p* < .05, \*\* *p* < .01, \*\*\* *p* < .001. Logit coefficients are log-odds; OR = exp(coef) where shown.
