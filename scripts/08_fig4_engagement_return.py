"""Figure 4: engagement return of indirect/mixed expression within peripheral discourse.

Reuses the H1 main spec (common.py), refits it, and produces *adjusted* predicted
log-engagement for Direct vs Indirect/mixed expression via marginal standardization
(g-computation), with account-clustered bootstrap 95% CIs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    PATH_ANALYSIS_ENTROPY,
    PROJECT_ROOT,
    apply_sample_mask,
    build_formula,
    fit_ols_cluster,
    prepare_model_frame,
    spec_by_model_id,
    utc_now_iso,
)

OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"

FOCAL = "indirect_clean"


def prepare_h1_frame(df: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    spec = spec_by_model_id("h1_engagement_indirect")
    sub = apply_sample_mask(df, spec.sample_key)
    sub["topic_id"] = sub["topic_id"].astype("Int64").astype(str)
    sub = prepare_model_frame(
        sub,
        y_col=spec.y_col,
        x_cols=list(spec.x_cols),
        control_cols=spec.control_cols,
        extra_cols=["topic_id"],
    )
    return spec, sub.reset_index(drop=True)


def cluster_bootstrap(
    sub: pd.DataFrame,
    x_full: np.ndarray,
    y_full: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    accounts = sub["account_id"].to_numpy()
    uniq = np.unique(accounts)
    pos = {a: np.where(accounts == a)[0] for a in uniq}
    rng = np.random.default_rng(seed)

    pred_a: list[float] = []
    pred_b: list[float] = []
    diff: list[float] = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([pos[a] for a in samp])
        beta_b, *_ = np.linalg.lstsq(x_full[sel], y_full[sel], rcond=None)
        m_a = float(x0[sel].mean(axis=0) @ beta_b)
        m_b = float(x1[sel].mean(axis=0) @ beta_b)
        pred_a.append(m_a)
        pred_b.append(m_b)
        diff.append(m_b - m_a)

    def ci(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {"direct": ci(pred_a), "indirect": ci(pred_b), "diff": ci(diff)}


def build_figure(
    points: dict[str, float],
    cis: dict[str, tuple[float, float]],
    beta: float,
    pvalue: float,
    pct_change: float,
) -> plt.Figure:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300}
    )
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    xs = [0, 1]
    ys = [points["direct"], points["indirect"]]
    yerr_low = [ys[0] - cis["direct"][0], ys[1] - cis["indirect"][0]]
    yerr_high = [cis["direct"][1] - ys[0], cis["indirect"][1] - ys[1]]

    ax.errorbar(
        xs,
        ys,
        yerr=[yerr_low, yerr_high],
        fmt="o",
        markersize=8,
        color="#33485c",
        ecolor="#33485c",
        elinewidth=1.4,
        capsize=6,
        capthick=1.4,
        zorder=3,
    )
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Direct\nexpression", "Indirect/mixed\nexpression"])
    ax.set_ylabel("Model-adjusted predicted log(1 + engagement)")

    span = (cis["indirect"][1] - cis["direct"][0]) or 1.0
    pad = span * 0.45
    ax.set_ylim(min(cis["direct"][0], cis["indirect"][0]) - pad, cis["indirect"][1] + pad)

    p_txt = "< .001" if pvalue < 0.001 else f"= {pvalue:.3f}"
    y_anno = cis["indirect"][1] + pad * 0.5
    ax.annotate(
        "Indirect/mixed − Direct:\n"
        rf"$\Delta$ = {beta:.3f}, $p$ {p_txt}"
        + f"\n$\\approx$ +{pct_change * 100:.1f}% in log-transformed engagement",
        xy=(0.5, y_anno),
        ha="center",
        va="center",
        fontsize=9.5,
        color="#16243a",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_ANALYSIS_ENTROPY)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str})
    if "verified" in df.columns:
        df["verified"] = df["verified"].fillna(0).astype(int)

    spec, sub = prepare_h1_frame(df)
    formula = build_formula(spec)

    fit, n_posts = fit_ols_cluster(sub, formula)
    beta = float(fit.params[FOCAL])
    pvalue = float(fit.pvalues[FOCAL])
    pct_change = float(np.exp(beta) - 1.0)
    n_accounts = int(sub["account_id"].nunique())

    # Adjusted predictions via g-computation on the actual covariate distribution.
    df0 = sub.copy()
    df0[FOCAL] = 0
    df1 = sub.copy()
    df1[FOCAL] = 1
    pred_direct = float(fit.predict(df0).mean())
    pred_indirect = float(fit.predict(df1).mean())

    # Precompute design matrices once for the fast cluster-bootstrap loop.
    y_full, x_full = patsy.dmatrices(formula, sub, return_type="dataframe")
    design_info = x_full.design_info
    x0 = patsy.build_design_matrices([design_info], df0, return_type="dataframe")[0].to_numpy()
    x1 = patsy.build_design_matrices([design_info], df1, return_type="dataframe")[0].to_numpy()
    cis = cluster_bootstrap(
        sub,
        x_full.to_numpy(),
        y_full.to_numpy().ravel(),
        x0,
        x1,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    # H1 uses PERIPHERAL_SAMPLE_CONTROLS (verified excluded by spec).
    control_note = list(spec.control_cols)

    ci_d = f"[{cis['direct'][0]:.4f}, {cis['direct'][1]:.4f}]"
    ci_i = f"[{cis['indirect'][0]:.4f}, {cis['indirect'][1]:.4f}]"
    ci_diff = f"[{cis['diff'][0]:.4f}, {cis['diff'][1]:.4f}]"
    diff_pt = pred_indirect - pred_direct

    points = {"direct": pred_direct, "indirect": pred_indirect}
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)

    fig = build_figure(points, cis, beta, pvalue, pct_change)
    png_path = OUT_FIGURES / "fig_4_engagement_return_indirect.png"
    pdf_path = OUT_FIGURES / "fig_4_engagement_return_indirect.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    pred_table = pd.DataFrame(
        [
            {
                "expression_type": "Direct expression",
                "predicted_log_engagement": round(pred_direct, 6),
                "ci95_low": round(cis["direct"][0], 6),
                "ci95_high": round(cis["direct"][1], 6),
                "n_posts": n_posts,
                "n_accounts": n_accounts,
            },
            {
                "expression_type": "Indirect/mixed expression",
                "predicted_log_engagement": round(pred_indirect, 6),
                "ci95_low": round(cis["indirect"][0], 6),
                "ci95_high": round(cis["indirect"][1], 6),
                "n_posts": n_posts,
                "n_accounts": n_accounts,
            },
        ]
    )
    csv_path = OUT_TABLES / "fig_4_adjusted_predictions.csv"
    pred_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    report = f"""# Figure 4 — Engagement return of indirect/mixed expression (report)

Built at: {utc_now_iso()}

## Data and sample
- Input file: `{inp.relative_to(PROJECT_ROOT)}`
- Sample filter (H1, reused from `common.apply_sample_mask("peripheral")`):
  `model_sample_h2` == True, i.e. `include_main == 1` AND `peripheral == 1`,
  with non-missing `log_engagement`, `indirect_clean`, `topic_id`, and all controls.
- Effective N (posts): **{n_posts:,}**
- Number of accounts (clusters): **{n_accounts:,}**

## Model
- Formula: `{formula}`
- Estimator: OLS with account-clustered robust SE (`cov_type="cluster"`, groups = `account_id`).
- Focal coefficient `indirect_clean` (0 = Direct, 1 = Indirect/mixed):
  - beta_indirect = **{beta:.4f}**
  - p-value = **{pvalue:.3g}**{" (< .001)" if pvalue < 0.001 else ""}
  - exp(beta) - 1 = **{pct_change * 100:.2f}%**

## Adjusted predictions (g-computation / marginal standardization)
Both scenarios use each post's actual covariates (T2, controls, topic FE); only
`indirect_clean` is set to 0 (Direct) vs 1 (Indirect/mixed), then predictions are averaged.

| Scenario | Adjusted predicted log engagement | 95% CI |
|----------|-----------------------------------|--------|
| Direct expression | {pred_direct:.4f} | {ci_d} |
| Indirect/mixed expression | {pred_indirect:.4f} | {ci_i} |
| Difference (Indirect − Direct) | {diff_pt:.4f} | {ci_diff} |

For a linear model with `indirect_clean` entering additively, the adjusted difference
equals beta_indirect by construction ({diff_pt:.4f} ≈ {beta:.4f}).

## Confidence-interval method
- **Cluster bootstrap = YES.** Resampling unit = `account_id` (not individual posts).
- Bootstrap replications: **{args.n_boot}** (seed = {args.seed}).
- Each replication: resample accounts with replacement, refit the H1 design via
  least squares, recompute the two scenario means and their difference; 95% CIs are
  percentile-based (2.5 / 97.5).

## Control specification
- H1 peripheral subsample uses `PERIPHERAL_SAMPLE_CONTROLS`: {control_note}
- `verified` is **excluded** by model spec (not retained as a zero-variance column).

## Outputs
- Figure: `outputs/figures/fig_4_engagement_return_indirect.png` / `.pdf`
- Table: `outputs/tables/fig_4_adjusted_predictions.csv`
"""
    report_path = OUT_REPORTS / "fig_4_engagement_return_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"figure -> {png_path}")
    print(f"figure -> {pdf_path}")
    print(f"table  -> {csv_path}")
    print(f"report -> {report_path}")
    print(
        f"N={n_posts}, accounts={n_accounts}, beta={beta:.4f}, p={pvalue:.3g}, "
        f"pct={pct_change * 100:.2f}%, controls={control_note}"
    )


if __name__ == "__main__":
    main()
