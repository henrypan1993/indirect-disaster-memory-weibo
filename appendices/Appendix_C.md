# Appendix C. Robustness to Alternative Peripheral Thresholds

This appendix evaluates the robustness of the peripheral-account definition.

Peripheral account position is defined as unverified accounts with follower counts below the 90th percentile of the Core corpus (N = 17,143). The percentile cut is computed once on that corpus (p90 ≈ 679,972). At the main cut, 10,877 posts are peripheral and 6,266 are non-peripheral. All models use account-clustered robust standard errors.

**Table C1. Sensitivity to alternative peripheral thresholds**

| Threshold | Follower cut (approx.) | H1: Indirect/Mixed → logged engagement | Peripheral → Indirect/Mixed (primary boundary model) |
|-----------|------------------------:|---------------------------------------:|-----------------------------------------------------:|
| p80 | 77,287 | 0.126*** (N = 10,650) | 0.804*** (OR ≈ 2.23) |
| p90 (main) | 679,972 | 0.132*** (N = 10,806) | 0.794*** (OR ≈ 2.21) |
| p95 | 1,967,472 | 0.130*** (N = 10,853) | 0.788*** (OR ≈ 2.20) |

*Notes.* H1 is estimated on the peripheral subsample with the main-text controls and topic fixed effects. The primary boundary model is the full-Core logit of Indirect/Mixed on peripheral status with period, posting features, and topic fixed effects, without verification status or continuous follower count. \*\*\* *p* < .001.

H1 remains positive and significant across p80–p95. The primary boundary model likewise remains positive and significant at all three cuts. The conditional boundary specification that additionally controls for verification and follower count is reported in Appendix D (Table D3).
