# Appendix D. Peripheral-Account Definition and Threshold Sensitivity

## D.1 Definition

$$
\mathrm{Peripheral}_i
=
\mathbf{1}\{\mathrm{verified}_i = 0 \;\wedge\; \mathrm{followers}_i < p90(\mathrm{followers})\}.
$$

The p90 threshold is computed once on the Gate-qualified Core corpus (N = 17,143 post records): **p90 = 679,971.6**. Post counts: peripheral = **10,877**; non-peripheral = **6,266**.

Peripheral status indexes **relative visibility position** (lower audience resources among non-verified accounts below a high follower percentile). It is not a measure of social marginality, political vulnerability, or algorithmic invisibility.

Counts above are **post-level**. Models use clustered robust standard errors by `account_id`.

## D.2 Why p90?

p90 balances overly narrow and overly broad upper-tail cuts. In this corpus, most accounts lie well below high percentiles, so peripheral post counts change only modestly across p80–p95.

## D.3 Threshold sensitivity

Exported sensitivity covers **p80, p90, and p95**. A p85 cut was not part of the archived analysis grid.

**Table D1. Peripheral threshold sensitivity (H1 and D3)**

Source: `outputs/public/table_peripheral_threshold_sensitivity.csv`.

| Threshold | Follower cut (approx.) | H1: indirect → log engagement | D3: peripheral → indirect |
|-----------|------------------------:|------------------------------:|--------------------------:|
| p80 | 77,287 | 0.126*** (N = 10,650) | 0.862* (OR ≈ 2.37) |
| p90 (main) | 679,972 | 0.132*** (N = 10,806) | 1.246 (OR ≈ 3.47, ns) |
| p95 | 1,967,472 | 0.130*** (N = 10,853) | Quasi-separation; not interpretable |

**Table D1 notes.** H1 remains positive and significant across thresholds. D3 is unstable (significant at p80, non-significant at p90, quasi-separated at p95) and is not treated as a robust alternative explanation for indirect expression.
