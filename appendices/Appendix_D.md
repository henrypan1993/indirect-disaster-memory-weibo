# Appendix D. Complete Regression Results

This appendix presents complete regression estimates for the principal models.

Account-clustered robust standard errors are reported in parentheses. \* *p* < .05, \*\* *p* < .01, \*\*\* *p* < .001. Topic fixed effects use a common reference category. Odds ratios (OR) equal exp(coef); OR confidence intervals equal exp(coef confidence intervals). Outcomes that enter models in logarithmic form are labeled as logged engagement (i.e., $\log(1+x)$).

## D.1 H1

The model specification follows the corresponding equation in the main text. The sample comprises peripheral posts with non-missing expression labels (N = 10,806; 9,897 accounts; R² = 0.111).

**Table D1. H1 OLS estimates (logged total engagement)**

| Variable | Coef. | SE | 95% CI |
|----------|------:|---:|--------|
| Intercept | 0.166* | 0.074 | [0.021, 0.311] |
| Indirect/Mixed | 0.132*** | 0.029 | [0.075, 0.190] |
| T2 | −0.181*** | 0.023 | [−0.226, −0.136] |
| log(followers) | 0.083*** | 0.005 | [0.073, 0.093] |
| Posting hour (CST) | 0.0004 | 0.001 | [−0.002, 0.003] |
| Hashtag count | 0.035** | 0.011 | [0.014, 0.056] |
| Text length | 0.0003*** | 0.00007 | [0.0002, 0.0004] |
| Topic fixed effects | Included | | |

*Component outcomes (same covariates).* Indirect/Mixed coefficients: logged likes 0.102*** (0.025), 95% CI [0.052, 0.152]; logged comments 0.070*** (0.020), 95% CI [0.031, 0.110]; logged reposts 0.007 (0.011), 95% CI [−0.015, 0.029].

## D.2 H2

The model specification follows the corresponding equation in the main text. The sample comprises disaster-impact-related posts from peripheral accounts with non-missing expression labels. The baseline specification uses N = 5,846. The composition-adjusted specification adds narrative type, dominant emotion, and relative window day (N = 5,804 after complete-case requirements).

**Table D2. H2 logit estimates (Indirect/Mixed)**

| Variable | Baseline | Composition-adjusted |
|----------|---------:|---------------------:|
| Intercept | −0.639* (0.266) | −0.602 (0.314) |
| T2 | 0.651*** (0.095) | 0.381** (0.116) |
| OR (T2) | 1.92 | 1.46 |
| OR 95% CI | [1.59, 2.31] | [1.17, 1.84] |
| log(followers) | 0.013 (0.018) | 0.002 (0.018) |
| Posting hour (CST) | −0.0003 (0.005) | −0.001 (0.006) |
| Hashtag count | 0.001 (0.030) | 0.007 (0.029) |
| Text length | −0.001* (0.001) | −0.001* (0.001) |
| Relative window day | — | −0.021* (0.010) |
| Topic fixed effects | Included | Included |
| Narrative fixed effects | Not included | Included |
| Emotion fixed effects | Not included | Included |
| N | 5,846 | 5,804 |

## D.3 Boundary test

The boundary-test specifications follow the corresponding equations in the main text. The sample is the full Core corpus with non-missing expression labels (N = 17,067; 14,138 accounts). The primary boundary model estimates the overall adjusted association of peripheral classification. The conditional boundary model estimates the association after additionally controlling for verification status and continuous follower count.

**Table D3. Boundary-test logit estimates (Indirect/Mixed)**

| Variable | Primary | Conditional |
|----------|--------:|------------:|
| Intercept | −2.488*** (0.172) | −2.701* (1.096) |
| Peripheral | 0.794*** (0.082) | 1.167 (1.075) |
| OR (Peripheral) | 2.21 | 3.21 |
| OR 95% CI | [1.88, 2.60] | [0.39, 26.42] |
| T2 | 0.506*** (0.071) | 0.483*** (0.072) |
| log(followers) | — | −0.032** (0.012) |
| Verified | — | 0.546 (1.073) |
| Posting hour (CST) | −0.007 (0.004) | −0.007 (0.004) |
| Hashtag count | −0.046 (0.031) | −0.038 (0.031) |
| Text length | −0.002* (0.001) | −0.002* (0.001) |
| Topic fixed effects | Included | Included |
| N | 17,067 | 17,067 |
