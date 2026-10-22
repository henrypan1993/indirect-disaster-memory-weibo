# Appendix B. Expression Coding and Validation

This appendix reports the operational definitions and validation of the expression annotation.

LLM labels are treated as fallible structured measurements. Annotation used a two-stage workflow: Gate screening for relevance, followed by Core coding of expression (and related fields) for relevant posts.

## B.1 Operational definitions

**Table B1. Expression labels**

| Label | Definition | Typical cues | Exclusions |
|-------|------------|--------------|------------|
| Direct | Single-layer, self-contained statement of fact, emotion, stance, memory, request, or information | Explicit “7·20” / flood naming; concrete help-seeking; clear emotion | Meaning that depends on allusion, irony, or multi-topic embedding |
| Indirect/Mixed | Meaning depends on allusion, metaphor, rhetorical question, comparison, sarcasm, visibility cues, or cross-event embedding | “again”, “same city”, rhetorical doubt, metaphor of unreceded water | Strong emotion alone; mixed emotion alone; slang alone |
| Unclear | Too fragmented to determine expression strategy | Broken fragments | Excluded from expression models |

**Modeling map:** Direct → 0; Indirect/Mixed → 1; Unclear → missing.

## B.2 Validation design

Primary validity compares Core LLM expression labels with a human-adjudicated gold set (199 effective pairs). On this set, expression agreement was moderate to substantial (κ = 0.581; accuracy = 0.879; macro-F1 = 0.764), while Indirect/Mixed recall was 0.500. Other annotation dimensions showed comparable moderate agreement.

## B.3 Confusion matrix

**Table B2. Expression confusion matrix (LLM vs gold, n = 199)**

| Gold \ LLM | Direct | Indirect/Mixed | Unclear | Row total |
|------------|-------:|---------------:|--------:|----------:|
| Direct | 154 | 5 | 0 | 159 |
| Indirect/Mixed | 17 | 18 | 1 | 36 |
| Unclear | 1 | 0 | 3 | 4 |

Indirect/Mixed recall = 18/36 = 0.500. Indirect/Mixed precision = 18/23 = 0.783. Macro-F1 is the unweighted mean of the three class F1 scores (Direct, Indirect/Mixed, Unclear) and equals 0.764. Overall accuracy is high because Direct dominates; the minority Indirect/Mixed class is under-detected and usually collapsed into Direct. Analyses therefore treat expression labels as fallible structured measurements.

## B.4 Wave-specific validation of expression labels

The same 199 pairs were split by retrieval wave. One T1 case had no gold expression label and is excluded here, as in Table B2. T1 (2021) contributes 117 pairs and T2 (2025) contributes 82. Gold Indirect/Mixed counts are balanced across waves (18 and 18).

**Table B3. Expression validation by period**

| Metric | 2021 | 2025 | Overall |
|--------|-----:|-----:|--------:|
| Human validation sample | 117 | 82 | 199 |
| Indirect/Mixed precision | 0.692 | 0.900 | 0.783 |
| Indirect/Mixed recall | 0.500 | 0.500 | 0.500 |
| Macro-F1 | 0.726 | 0.791 | 0.764 |

Precision and recall are for the Indirect/Mixed class, with the human gold label as the reference. Macro-F1 uses the same three-class definition as the pooled estimate.

**Table B4. Expression confusion matrix, 2021 (n = 117)**

| Gold \ LLM | Direct | Indirect/Mixed | Unclear | Row total |
|------------|-------:|---------------:|--------:|----------:|
| Direct | 93 | 4 | 0 | 97 |
| Indirect/Mixed | 9 | 9 | 0 | 18 |
| Unclear | 1 | 0 | 1 | 2 |
| Column total | 103 | 13 | 1 | 117 |

Indirect/Mixed precision = 9/13 = 0.692. Indirect/Mixed recall = 9/18 = 0.500.

**Table B5. Expression confusion matrix, 2025 (n = 82)**

| Gold \ LLM | Direct | Indirect/Mixed | Unclear | Row total |
|------------|-------:|---------------:|--------:|----------:|
| Direct | 61 | 1 | 0 | 62 |
| Indirect/Mixed | 8 | 9 | 1 | 18 |
| Unclear | 0 | 0 | 2 | 2 |
| Column total | 69 | 10 | 3 | 82 |

Indirect/Mixed precision = 9/10 = 0.900. Indirect/Mixed recall = 9/18 = 0.500.

The recall of 0.500 is not an average of unequal period-specific recalls: each wave misses half of its 18 gold Indirect/Mixed posts, and nearly all of those misses are labeled Direct (9 in 2021; 8 in 2025, plus 1 labeled Unclear). Precision is lower in 2021 because 4 gold Direct posts are called Indirect/Mixed, compared with 1 such case in 2025. The 2025 wave therefore does not show greater over-detection of Indirect/Mixed. Cell counts are small, especially for Unclear and for the precision contrast, so Table B3 is a descriptive check of whether the pooled error pattern is period-specific. It is not a formal test of differential measurement error. The shared error—under-detection of Indirect/Mixed, usually as Direct—is present in both waves.

## B.5 Misclassification sensitivity analysis

### B.5.1 Misclassification parameters

This analysis assesses the sensitivity of the observed period difference in Indirect/Mixed Expression to classification errors. The descriptive contrast is 14.0% (675/4,827) in 2021 and 29.4% (300/1,019) in 2025 among disaster-impact-related posts from peripheral accounts. The primary composition-adjusted logistic model (repository `h2_m2_composition`, N = 5,804) yielded an odds ratio of 1.46 (95% CI [1.17, 1.84]).

Wave-specific sensitivity and false-positive rates were estimated from the human-adjudicated validation set. Because the primary binary expression models excluded Unclear labels, pairs whose gold or LLM expression was Unclear were excluded from the binary rate calculations. Estimated sensitivity was 0.500 (9/18) in 2021 and 0.529 (9/17) in 2025. Estimated false-positive rates were 0.041 (4/97) and 0.016 (1/62), respectively.

### B.5.2 Sensitivity scenarios

Let π be the true Indirect/Mixed share and p the observed share. With period-specific sensitivity Se and false-positive rate FPR,

p = FPR + (Se − FPR) π,

so π = (p − FPR) / (Se − FPR) when Se > FPR (Rogan–Gladen correction). Corrected proportions were calculated under wave-specific validation rates, pooled rates (Se = 18/35 = 0.514, FPR = 5/159 = 0.031), and an assumed sensitivity of 0.80 with the validation false-positive rates held fixed.

**Table B6. Sensitivity of period differences to expression-label misclassification**

| Scenario | 2021 | 2025 | Difference (pp) | Crude OR | Adjusted OR |
|----------|-----:|-----:|----------------:|---------:|------------:|
| Observed LLM labels | 14.0% | 29.4% | 15.5 | 2.57 | 1.46 |
| Wave-specific error rates | 21.5% | 54.2% | 32.7 | 4.33 | 3.39 |
| Pooled error rates | 22.4% | 54.5% | 32.0 | 4.13 | 3.16 |
| Assumed sensitivity = 0.80 | 13.0% | 35.5% | 22.5 | 3.69 | 2.54 |

Percentage-point differences use unrounded estimates. Adjusted odds ratios for the sensitivity scenarios are medians from latent-label imputation and refitting of `h2_m2_composition`. The observed adjusted odds ratio of 1.46 is the primary estimate, not an imputed median.

### B.5.3 Uncertainty in corrected estimates

For each imputed data set, a latent Indirect/Mixed label was drawn from the period-specific positive and negative predictive probabilities implied by the Rogan–Gladen correction, and the composition-adjusted logit was refit. The random seed is `20261022`. With wave-specific rates held fixed, 80 imputations yielded a median odds ratio of 3.39 (range 2.99–3.81), and every draw exceeded 1. The pooled-rate median was 3.16, and the sensitivity-0.80 median was 2.54. Those ranges reflect imputation under fixed error rates. They do not include uncertainty in the error rates themselves.

Validation pairs were then resampled with replacement within each wave, and one imputation was drawn for each resample (`N_BOOT` = 200). Of those draws, 192 produced a finite odds ratio. The median was 3.29; 94.8% exceeded 1; the 2.5th–97.5th percentile range was 0.69–27.8. Holding the other estimated parameters fixed, the 2025 false-positive rate would need to rise to about 0.230 before the corrected prevalence difference disappeared. That is substantially higher than the estimated 2025 false-positive rate of 0.016.

### B.5.4 Interpretation and limitations

Across the examined scenarios, the corrected prevalence differences and adjusted odds ratios were larger than the observed estimates. This pattern is consistent with attenuation of the period association under the specified misclassification mechanism.

The validation sample was enriched with difficult-to-classify cases, and the wave-specific rates come from small cells, so their applicability to the main analytical sample is uncertain. The binary correction also drops pairs that involve Unclear. Resampling produced corrected odds ratios below 1 as well as very large odds ratios. The sensitivity analysis supports the plausibility of attenuation. It does not establish the magnitude of measurement bias, and it does not replace the observed 14.0% to 29.4% contrast or the odds ratio of 1.46.

Aggregate outputs are `outputs/public/table_expression_validation_period_rates.csv`, `table_misclassification_sensitivity_shares.csv`, and `table_misclassification_sensitivity_or.csv`. `scripts/14_misclassification_sensitivity.py` recomputes them from `data/restricted/expression_validation_pairs.csv` when that restricted file is available. The row-level validation file is not part of the public tree. The published H2 counts used for the share correction are in `data/public/h2_indirect_counts.csv`. Refitting the imputed odds ratios also requires `data/processed/model_data_final.csv`.
