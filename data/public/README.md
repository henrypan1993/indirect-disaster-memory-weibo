# Public aggregates for Appendix B

This directory contains aggregate counts only. It has no post text, `mid`, account identifier, or row-level validation pair.

| File | Contents |
|------|----------|
| `h2_indirect_counts.csv` | H2 descriptive sample sizes and Indirect/Mixed counts (675/4,827 in 2021; 300/1,019 in 2025) |

`scripts/14_misclassification_sensitivity.py` uses this file for the share correction. Recomputing sensitivity and false-positive rates requires the restricted validation file, which is not in this directory. Re-fitting the imputed odds ratio also requires the restricted model table.
