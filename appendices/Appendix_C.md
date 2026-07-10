# Appendix C. Topic Entropy Construction and Robustness

Topic entropy is a **secondary / diagnostic** measure (formal label **D1**). It is not a core hypothesis equal to H1/H2.

## C.1 Construction

Unique texts among `include_main == 1` posts (N = 13,508 by `text_hash`) were embedded with `shibing624/text2vec-base-chinese`, clustered with K-Means (**K = 10**), converted to soft probabilities via temperature softmax (**τ = 0.10**), and scored with normalized Shannon entropy:

$$
\mathrm{Entropy}_i
=
\frac{-\sum_{k=1}^{K} p_{ik}\log p_{ik}}{\log K}
\in [0,1].
$$

Hard topic assignment `topic_id = argmax_k p_{ik}` supplies topic fixed effects in engagement models. Mean entropy ≈ 0.910 (min ≈ 0.569, max ≈ 0.988).

## C.2 Secondary analysis (D1)

Among peripheral trauma-related posts, T2 is associated with slightly lower entropy (β ≈ −0.010, *p* < .001). The absolute magnitude is small; interpretation is limited semantic narrowing, not dramatic topical collapse.

## C.3 Robustness across K and τ

**Table C1. Entropy and related focal coefficients across K × τ**

Source (long format): `outputs/public/table_entropy_ktau_grid.csv`.

**CSV structure.** The file is long-format, not one row per grid cell:

- **D1:** 9 rows = one `t2` coefficient for each of $K\in\{8,10,12\}\times\tau\in\{0.05,0.10,0.20\}$ (`spec_id` such as `k8_tau005`, `k10_tau010`, …).
- **D2:** 18 rows = for each of the same 9 cells, two terms (`entropy_norm` and `indirect_clean`).
- Total **27 rows**. The main cell (`k10_tau010`) is merged from the main-model summary because the robustness export omitted the main K/τ pair.

| K | τ | D1: T2 → entropy | D2: entropy → engagement | D2: indirect → engagement |
|---|----|------------------:|-------------------------:|--------------------------:|
| 8 | 0.05 | −0.015*** | 0.051 | 0.138*** |
| 8 | 0.10 | −0.008*** | −0.020 | 0.139*** |
| 8 | 0.20 | −0.003*** | −0.304 | 0.139*** |
| 10 | 0.05 | −0.015*** | 0.018 | 0.132*** |
| **10** | **0.10 (main)** | **−0.010*** | **−0.090** | **0.132*** |
| 10 | 0.20 | −0.004*** | −0.806 | 0.133*** |
| 12 | 0.05 | 0.000 (*p* = .994) | 0.140 | 0.128*** |
| 12 | 0.10 | −0.003* | 0.195 | 0.128*** |
| 12 | 0.20 | −0.002*** | −0.223 | 0.129*** |

**Table C1 notes.** Direction of D1 is negative in 8 of 9 cells; magnitude is τ-sensitive. Entropy never significantly predicts engagement; the indirect → engagement association remains stable.

## C.4 Diagnostic framing

Because entropy depends on embedding and clustering choices, and because effects are small and τ-sensitive, D1 is reported as a diagnostic secondary analysis supporting limited semantic concentration alongside the primary expression-form shift (H2).
