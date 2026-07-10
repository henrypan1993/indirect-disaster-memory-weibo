# Model name mapping

Public manuscript labels versus repository `model_id` strings (do not rename scripts in v1):

| Formal label | `model_id` | Role |
|--------------|------------|------|
| H1 | `h1_engagement_indirect` | Engagement ~ indirect/mixed (peripheral subsample) |
| H2 | `h2b_indirect_reactivation` | Indirect/mixed ~ T2 (peripheral trauma-related subsample) |
| D1 | `h2a_entropy_reactivation` | Entropy ~ T2 (secondary / diagnostic) |
| D2 | `e2_entropy_increment` | Engagement ~ indirect + entropy (boundary) |
| D3 | `e1b_indirect_peripheral` | Indirect/mixed ~ peripheral (boundary) |

## Excluded legacy outputs

Not part of the public manuscript mapping and not published under `outputs/public/`:

- `e1a_entropy_peripheral` (historical account-position → entropy run; outside final H1/H2/D1–D3 structure)
- Legacy filenames such as `h1_entropy_model`, `h3_*`, `h4_*`
- Pre-fix archives under `outputs/models/_archive_pre_control_fix/`
