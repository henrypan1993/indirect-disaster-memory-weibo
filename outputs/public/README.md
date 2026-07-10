# Public curated coefficient tables

These files are **whitelist-filtered** to the final manuscript models only:

| formal_label | model_id |
|--------------|----------|
| H1 | `h1_engagement_indirect` |
| H2 | `h2b_indirect_reactivation` |
| D1 | `h2a_entropy_reactivation` |
| D2 | `e2_entropy_increment` |
| D3 | `e1b_indirect_peripheral` |

Named files:

| File | Contents |
|------|----------|
| `table_main_focal_coefficients.csv` | Focal predictors only (6 rows) |
| `table_main_selected_coefficients.csv` | Focal + `t2` control terms from the upstream “full” export (9 rows; **not** every control/FE coefficient) |
| `table_h1_engagement_components.csv` | Likes / comments / reposts |
| `table_robustness_focal.csv` | Whitelist robustness focal rows |
| `table_peripheral_threshold_sensitivity.csv` | H1 + D3 × p80/p90/p95 |
| `table_entropy_ktau_grid.csv` | Long-format K×τ grid: 9 D1 + 18 D2 = 27 rows (includes main `k10_tau010`) |
| `summary_main_models.csv` / `summary_robustness_models.csv` | Whitelist summaries |

Original pipeline outputs under `outputs/tables/` and `outputs/models/` are left unchanged and are not published when they contain other models.

See `MANIFEST.csv` for row counts and source paths.
