# Appendix E. Full Regression Tables

Clustered robust standard errors by `account_id` in parentheses. Peripheral-subsample models omit `verified` (structurally constant). Topic fixed effects use hard-assigned `topic_id` (reference = topic 0).

Curated focal coefficients: `outputs/public/table_main_focal_coefficients.csv`. Full text dumps: selected files under `outputs/models/`.

## E.1 Model overview

**Table E1. Analytic samples and fit**

| Label | Outcome | Focal predictor(s) | Sample | N posts | N accounts | R² / pseudo-R² |
|-------|---------|--------------------|--------|--------:|-----------:|----------------|
| H1 | `log_engagement` | Indirect/mixed | Peripheral | 10,806 | 9,897 | R² = 0.111 |
| H2 | Indirect/mixed | T2 | Peripheral trauma-related | 5,846 | 5,476 | Pseudo-R² (CS) = 0.026 |
| D1 | `entropy_norm` | T2 | Peripheral trauma-related | 5,880 | 5,508 | R² = 0.020 |
| D2 | `log_engagement` | Indirect/mixed; entropy | Peripheral | 10,806 | 9,897 | R² = 0.111 |
| D3 | Indirect/mixed | Peripheral | Full Core (valid expression) | 17,067 | 14,138 | Pseudo-R² (CS) = 0.036 |

## E.2 H1 — Engagement and indirect/mixed expression

$$
\log(1+\mathrm{Engagement}_i)
=
\alpha + \beta_1 \mathrm{Indirect/Mixed}_i + \lambda \mathrm{T2}_i + \mathbf{X}_i\boldsymbol{\gamma} + \delta_{z_i} + \varepsilon_i.
$$

**Table E2. H1 OLS coefficients (selected)**

| Variable | Coef. | SE |
|----------|------:|---:|
| Indirect/mixed | 0.132*** | 0.029 |
| T2 | −0.181*** | 0.023 |
| log(followers) | 0.083*** | 0.005 |
| Posting hour (CST) | 0.0004 | 0.001 |
| Hashtag count | 0.035*** | 0.011 |
| Text length | 0.0003*** | 0.00007 |
| Topic FE | Yes | |
| N / R² | 10,806 / 0.111 | |

Source file: `outputs/models/h1_engagement_indirect.txt`. Approx. %Δ for indirect/mixed: +14.1%.

## E.3 H1 engagement components (robustness)

**Table E3. Indirect/mixed coefficients by engagement component**

| Outcome | Coef. | SE | Approx. %Δ |
|---------|------:|---:|-----------:|
| log(likes) | 0.102*** | 0.025 | +10.7% |
| log(comments) | 0.070*** | 0.020 | +7.3% |
| log(reposts) | 0.007 | 0.011 | +0.7% |

Source: `outputs/public/table_h1_engagement_components.csv`.

## E.4 H2 — Indirect/mixed expression and T2

$$
\operatorname{logit}\!\left[\Pr(\mathrm{Indirect/Mixed}_i=1)\right]
=
\alpha + \beta \mathrm{T2}_i + \mathbf{X}_i\boldsymbol{\gamma}.
$$

**Table E4. H2 logit coefficients (selected)**

| Variable | Coef. | SE | OR |
|----------|------:|---:|---:|
| T2 | 0.880*** | 0.083 | **2.41** |
| log(followers) | 0.020 | 0.017 | — |
| Hashtag count | −0.089** | 0.027 | — |
| Text length | −0.001* | 0.0005 | — |
| N / Pseudo-R² (CS) | 5,846 / 0.026 | | |

Source file: `outputs/models/h2b_indirect_reactivation.txt`.

## E.5 D1–D3 — Secondary and boundary models

**Table E5. Focal coefficients for D1–D3**

| Label | Focal term | Coef. | SE | *p* | Notes |
|-------|------------|------:|---:|----:|-------|
| D1 | T2 → entropy | −0.010*** | 0.001 | <.001 | Secondary / diagnostic |
| D2 | Indirect → engagement | 0.132*** | 0.029 | <.001 | Stable vs H1 |
| D2 | Entropy → engagement | −0.090 | 0.248 | .716 | Does not explain H1 |
| D3 | Peripheral → indirect | 1.246 | 1.042 | .232 | OR ≈ 3.47; ns |

Sources: `outputs/models/h2a_entropy_reactivation.txt`, `e2_entropy_increment.txt`, `e1b_indirect_peripheral.txt`; curated focal CSV.
