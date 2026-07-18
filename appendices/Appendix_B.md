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

Indirect/Mixed recall = 18/36 = 0.500. Overall accuracy is high because Direct dominates; the minority Indirect/Mixed class is under-detected and usually collapsed into Direct. Analyses therefore treat expression labels as fallible structured measurements.
