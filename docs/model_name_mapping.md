# Model name mapping

Public manuscript labels versus repository `model_id` / `spec_id`:

| Formal label | `model_id` | Primary `spec_id` | Role |
|--------------|------------|-------------------|------|
| H1 | `h1_engagement_indirect` | `main` | Engagement ~ indirect/mixed (peripheral subsample) |
| H2 | `h2_indirect_period` | `h2_m1_period` | Indirect/mixed ~ T2 (disaster-impact-related peripheral subsample) |
| D1 | `d1_indirect_peripheral` | `d1_m1_total_association` | Indirect/mixed ~ peripheral (full Core; total association) |

## Spec variants (same formal label)

| Formal | `spec_id` | Role |
|--------|-----------|------|
| H2 | `h2_m1_period` | **Main:** period association with standard controls + topic FE |
| H2 | `h2_m2_composition` | Composition adjustment (+ narrative, emotion, relative window day) |
| D1 | `d1_m1_total_association` | **Main diagnostic:** peripheral without verified/log_followers |
| D1 | `d1_m2_conditional_threshold` | Conditional: adds verified + log_followers |

## Former `model_id` aliases

| Current `model_id` | Former `model_id` |
|--------------------|-------------------|
| `h2_indirect_period` | `h2b_indirect_reactivation` |
| `d1_indirect_peripheral` | `e1b_indirect_peripheral` |

Sample flag: `disaster_impact_related` (alias of former `narrative_trauma_clean`).
