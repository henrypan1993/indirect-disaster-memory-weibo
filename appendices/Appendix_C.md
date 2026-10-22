# Appendix C. Robustness to Alternative Peripheral Thresholds

This appendix evaluates the robustness of the peripheral-account definition.

Peripheral account position is defined as unverified accounts with follower counts below the 90th percentile of the Core corpus (N = 17,143). The percentile cut is computed once on that corpus (p90 ≈ 679,972). At the main cut, 10,877 posts are peripheral and 6,266 are non-peripheral. All models use account-clustered robust standard errors.

**Table C1. Sensitivity to alternative peripheral thresholds**

| Threshold | Follower cut (approx.) | Indirect/Mixed → logged engagement | Peripheral → Indirect/Mixed (primary boundary model) |
|-----------|------------------------:|-----------------------------------:|-----------------------------------------------------:|
| p80 | 77,287 | 0.126*** (N = 10,650) | 0.804*** (OR ≈ 2.23) |
| p90 (main) | 679,972 | 0.132*** (N = 10,806) | 0.794*** (OR ≈ 2.21) |
| p95 | 1,967,472 | 0.130*** (N = 10,853) | 0.788*** (OR ≈ 2.20) |

*Notes.* The engagement column is the broader peripheral sample (any narrative, non-missing expression), the same population as manuscript Q2 and Appendix D.1. Curated rows: `outputs/public/table_q2_peripheral_thresholds_broader.csv`. The p90 row is the same specification as `h1_broader_sample`. The primary boundary model is the full-Core logit of Indirect/Mixed on peripheral status with period, posting features, and topic fixed effects, without verification status or continuous follower count (`outputs/public/table_peripheral_threshold_sensitivity.csv`, D1 rows). \*\*\* *p* < .001.

Indirect/Mixed → logged engagement remains positive and significant across p80–p95. The primary boundary model likewise remains positive and significant at all three cuts. The conditional boundary specification that additionally controls for verification and follower count is reported in Appendix D (Table D3).

A separate repository check refits the engagement equation on the disaster-impact × peripheral core sample at the same cuts (N = 5,786 / 5,846 / 5,863; coefficients 0.114** / 0.123*** / 0.123***). Those rows are in `table_peripheral_threshold_sensitivity.csv` under `h1_engagement_indirect`. They are not Table C1 and they are not the manuscript’s primary Q2 estimate.
