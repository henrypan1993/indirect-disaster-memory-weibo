# Appendices (English)

Formal English appendices for the manuscript. One canonical file per appendix.

| File | Role |
|------|------|
| [Appendix_A.md](Appendix_A.md) | Data retrieval and corpus construction |
| [Appendix_B.md](Appendix_B.md) | Codebook, annotation, and validation (Tables B1–B3) |
| [Appendix_C.md](Appendix_C.md) | Topic entropy construction and robustness |
| [Appendix_D.md](Appendix_D.md) | Peripheral definition and threshold sensitivity |
| [Appendix_E.md](Appendix_E.md) | Full regression tables |

## Formal labels

| Label | Meaning |
|-------|---------|
| H1 | Engagement ~ indirect/mixed (peripheral subsample) |
| H2 | Indirect/mixed ~ T2 (peripheral trauma-related subsample) |
| D1 | Entropy ~ T2 (secondary / diagnostic) |
| D2 | Engagement ~ indirect + entropy (boundary) |
| D3 | Indirect/mixed ~ peripheral (boundary) |

Repository `model_id` strings are listed only in [`docs/model_name_mapping.md`](../docs/model_name_mapping.md).

## Statistical conventions

Clustered robust standard errors by `account_id`. \* *p* < .05, \*\* *p* < .01, \*\*\* *p* < .001. Logit coefficients are log-odds; OR = exp(coef) where shown.
