# Appendix B. Codebook, Annotation Workflow, and Validation

LLM labels are treated as **fallible structured measurements**. Production prompts reside in the upstream labeling project; this appendix records the coding rules and validation metrics used in the manuscript.

## B.1 Two-stage workflow

1. **Gate** — relevance and analytical-value screening; clarity recorded for diagnostics.
2. **Core** — narrative, emotion, and expression coding for posts with `label_relevance = 1` and `score_relevance = 2`.

## B.2 Expression-form definitions

**Table B1. Expression labels**

| Label | Definition | Typical cues | Exclusions |
|-------|------------|--------------|------------|
| Direct | Single-layer, self-contained statement of fact, emotion, stance, memory, request, or information | Explicit “7·20” / flood naming; concrete help-seeking; clear emotion | Meaning that depends on allusion, irony, or multi-topic embedding |
| Indirect-Mixed | Meaning depends on allusion, metaphor, rhetorical question, comparison, sarcasm, visibility cues, or cross-event embedding | “again”, “same city”, rhetorical doubt, metaphor of unreceded water | Strong emotion alone; mixed emotion alone; slang alone |
| Unclear | Too fragmented to determine expression strategy | Broken fragments | Excluded from expression models (`indirect_clean` missing) |

**Modeling map:** Direct → 0; Indirect-Mixed → 1; Unclear → NA.

## B.3 Structured outputs

Gate JSON keys: `mid`, `label_relevance`, `score_relevance`, `score_clarity`, reasons.  
Core JSON keys: `mid`, `label_narrative`, `label_emotion`, `label_expression`, reasons. Allowed expression set: `{Direct, Indirect-Mixed, Unclear}`.

## B.4 Validation procedure

Primary validity: Core LLM versus human-adjudicated gold (core_B200; 199 effective pairs). Auxiliary Gate checks use pilot joint-human labels. Post-annotation QA (n = 40) is a stability check, not co-equal primary evidence.

## B.5 Validation metrics

**Table B2. Agreement and classification metrics**

| Stage | Field | Comparison | n | κ / κw | Additional |
|-------|-------|------------|--:|--------|------------|
| Gate | Relevance label | LLM vs joint human (P100) | 100 | 0.637 | Acc = 0.910; F1 = 0.947 |
| Gate | Relevance score | LLM vs joint human (P100) | 100 | 0.639 (κw) | MAE = 0.290 |
| Gate | Clarity score | LLM vs joint human (P100) | 81 | 0.649 (κw) | MAE = 0.136 |
| Core | Narrative | LLM vs gold | 199 | 0.580 | Acc = 0.693; Macro-F1 = 0.631 |
| Core | Emotion | LLM vs gold | 199 | 0.539 | Acc = 0.638; Macro-F1 = 0.570 |
| Core | Expression | LLM vs gold | 199 | 0.581 | Acc = 0.879; Macro-F1 = 0.764; Indirect-Mixed recall = 0.500 |

**Table B3. Expression confusion matrix (LLM vs gold, n = 199)**

| Gold \ LLM | Direct | Indirect-Mixed | Unclear | Row total |
|------------|-------:|---------------:|--------:|----------:|
| Direct | 154 | 5 | 0 | 159 |
| Indirect-Mixed | 17 | 18 | 1 | 36 |
| Unclear | 1 | 0 | 3 | 4 |

Indirect-Mixed recall = 18/36 = 0.500. Overall expression accuracy is high because Direct dominates; the minority Indirect-Mixed class is under-detected, usually collapsed into Direct. Analyses therefore treat expression labels as noisy structured measurements.
