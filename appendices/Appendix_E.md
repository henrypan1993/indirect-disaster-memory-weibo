# Appendix E. Full Regression Tables

Clustered robust standard errors by `account_id` in parentheses. \* *p* < .05, \*\* *p* < .01, \*\*\* *p* < .001. Peripheral-subsample models omit `verified` (structurally constant). Topic fixed effects use hard-assigned `topic_id` (reference = topic 0). Machine-readable dumps: `outputs/models/*.txt`.

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
\log(1+\mathrm{Engagement}_i) = \alpha + \beta_1 \mathrm{Indirect/Mixed}_i + \lambda \mathrm{T2}_i + \mathbf{X}_i\boldsymbol{\gamma} + \delta_{z_i} + \varepsilon_i.
$$

**Table E2. H1 OLS (full coefficients)**

Source: `outputs/models/h1_engagement_indirect.txt`.

| Variable | Coef. | SE |
|----------|------:|---:|
| Intercept | 0.166* | 0.074 |
| Topic 1 | 0.200** | 0.069 |
| Topic 2 | −0.427*** | 0.070 |
| Topic 3 | 0.113 | 0.072 |
| Topic 4 | −0.341*** | 0.101 |
| Topic 5 | −0.210** | 0.070 |
| Topic 6 | −0.445*** | 0.096 |
| Topic 7 | −0.367*** | 0.099 |
| Topic 8 | −0.158* | 0.068 |
| Topic 9 | −0.088 | 0.067 |
| Indirect/mixed | 0.132*** | 0.029 |
| T2 | −0.181*** | 0.023 |
| log(followers) | 0.083*** | 0.005 |
| Posting hour (CST) | 0.0004 | 0.001 |
| Hashtag count | 0.035*** | 0.011 |
| Text length | 0.0003*** | 0.00007 |
| N / R² / Adj. R² | 10,806 / 0.111 / 0.110 | |

Approx. %Δ for indirect/mixed: +14.1%.

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
\operatorname{logit}\!\left[\Pr(\mathrm{Indirect/Mixed}_i=1)\right] = \alpha + \beta \mathrm{T2}_i + \mathbf{X}_i\boldsymbol{\gamma}.
$$

**Table E4. H2 logit (full coefficients)**

Source: `outputs/models/h2b_indirect_reactivation.txt`.

| Variable | Coef. | SE | OR |
|----------|------:|---:|---:|
| Intercept | −1.714*** | 0.126 | — |
| T2 | 0.880*** | 0.083 | 2.41 |
| log(followers) | 0.020 | 0.017 | — |
| Posting hour (CST) | 0.001 | 0.005 | — |
| Hashtag count | −0.089** | 0.027 | — |
| Text length | −0.001* | 0.0005 | — |
| N / Pseudo-R² (CS) | 5,846 / 0.026 | | |

## E.5 D1 — Topic entropy and T2

$$
\mathrm{Entropy}_i = \alpha + \beta \mathrm{T2}_i + \mathbf{X}_i\boldsymbol{\gamma} + \varepsilon_i.
$$

**Table E5. D1 OLS (full coefficients)**

Source: `outputs/models/h2a_entropy_reactivation.txt`.

| Variable | Coef. | SE |
|----------|------:|---:|
| Intercept | 0.922*** | 0.002 |
| T2 | −0.010*** | 0.001 |
| log(followers) | −0.0002 | 0.0003 |
| Posting hour (CST) | −0.0002* | 0.00007 |
| Hashtag count | 0.002*** | 0.0004 |
| Text length | −0.000012*** | 0.000003 |
| N / R² / Adj. R² | 5,880 / 0.020 / 0.019 | |

## E.6 D2 — Engagement with entropy increment

$$
\log(1+\mathrm{Engagement}_i) = \alpha + \beta_1 \mathrm{Indirect/Mixed}_i + \beta_2 \mathrm{Entropy}_i + \lambda \mathrm{T2}_i + \mathbf{X}_i\boldsymbol{\gamma} + \delta_{z_i} + \varepsilon_i.
$$

**Table E6. D2 OLS (full coefficients)**

Source: `outputs/models/e2_entropy_increment.txt`.

| Variable | Coef. | SE |
|----------|------:|---:|
| Intercept | 0.245 | 0.230 |
| Topic 1 | 0.206** | 0.071 |
| Topic 2 | −0.421*** | 0.072 |
| Topic 3 | 0.117 | 0.073 |
| Topic 4 | −0.347*** | 0.105 |
| Topic 5 | −0.210** | 0.070 |
| Topic 6 | −0.443*** | 0.097 |
| Topic 7 | −0.376*** | 0.101 |
| Topic 8 | −0.151* | 0.072 |
| Topic 9 | −0.086 | 0.068 |
| Indirect/mixed | 0.132*** | 0.029 |
| Entropy (normalized) | −0.090 | 0.248 |
| T2 | −0.183*** | 0.023 |
| log(followers) | 0.083*** | 0.005 |
| Posting hour (CST) | 0.0004 | 0.001 |
| Hashtag count | 0.035*** | 0.011 |
| Text length | 0.0003*** | 0.00007 |
| N / R² / Adj. R² | 10,806 / 0.111 / 0.110 | |

## E.7 D3 — Indirect/mixed expression and peripheral position

$$
\operatorname{logit}\!\left[\Pr(\mathrm{Indirect/Mixed}_i=1)\right] = \alpha + \beta \mathrm{Peripheral}_i + \lambda \mathrm{T2}_i + \mathbf{X}_i\boldsymbol{\gamma}.
$$

**Table E7. D3 logit (full coefficients)**

Source: `outputs/models/e1b_indirect_peripheral.txt`.

| Variable | Coef. | SE | OR |
|----------|------:|---:|---:|
| Intercept | −2.807** | 1.052 | — |
| Peripheral | 1.246 | 1.042 | 3.47 |
| T2 | 0.622*** | 0.060 | — |
| log(followers) | −0.030* | 0.012 | — |
| Verified | 0.448 | 1.039 | — |
| Posting hour (CST) | −0.001 | 0.004 | — |
| Hashtag count | −0.191*** | 0.032 | — |
| Text length | −0.003*** | 0.001 | — |
| N / Pseudo-R² (CS) | 17,067 / 0.036 | | |
