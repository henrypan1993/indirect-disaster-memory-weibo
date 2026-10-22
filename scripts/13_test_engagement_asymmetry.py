"""Formal test of engagement-component asymmetry (Indirect/Mixed).

Separate OLS already shows Indirect/Mixed → likes/comments significant and
reposts not. That pattern alone does not prove coefficient differences.
This script stacks the three component equations on the H1 core sample and
tests equality of the Indirect/Mixed coefficients with account-clustered SEs.

Method: stacked OLS with outcome-type × predictors (reference = reposts).
Wald tests: joint β_L=β_C=β_R; pairwise L−R, C−R, L−C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    PATH_MODEL_DATA_FINAL,
    PERIPHERAL_SAMPLE_CONTROLS,
    PROJECT_ROOT,
    REF_TOPIC,
    ZeroVarianceDiagnosticError,
    apply_sample_mask,
    diagnose_model_frame,
    fit_ols_cluster,
    format_diagnostic_header,
    prepare_model_frame,
    utc_now_iso,
    write_json_report,
)

OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"
OUT_PUBLIC = PROJECT_ROOT / "outputs" / "public"

OUTCOME_COLS: tuple[tuple[str, str], ...] = (
    ("likes", "log_likes"),
    ("comments", "log_comments"),
    ("reposts", "log_reposts"),
)
REF_OUTCOME = "reposts"
FOCAL = "indirect_clean"


def stacked_rhs_formula() -> str:
    """RHS interacted with outcome type; reference outcome = reposts."""
    parts = [
        FOCAL,
        "t2",
        *PERIPHERAL_SAMPLE_CONTROLS,
        f"C(topic_id, Treatment(reference='{REF_TOPIC}'))",
    ]
    predictors = " + ".join(parts)
    return f"C(outcome, Treatment(reference='{REF_OUTCOME}')) * ({predictors})"


def stacked_formula() -> str:
    return f"y ~ {stacked_rhs_formula()}"


def stack_engagement_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Long stack of likes / comments / reposts on a common covariate frame."""
    parts: list[pd.DataFrame] = []
    for outcome, y_col in OUTCOME_COLS:
        if y_col not in df.columns:
            msg = f"missing outcome column: {y_col}"
            raise ValueError(msg)
        part = df.copy()
        part["y"] = pd.to_numeric(part[y_col], errors="coerce")
        part["outcome"] = outcome
        parts.append(part)
    stacked = pd.concat(parts, ignore_index=True)
    stacked = stacked.dropna(subset=["y", "account_id"])
    return stacked


def _param_containing(index: pd.Index, *needles: str) -> str:
    hits = [str(p) for p in index if all(n in str(p) for n in needles)]
    if len(hits) != 1:
        msg = f"expected one param containing {needles!r}, found {hits!r}"
        raise KeyError(msg)
    return hits[0]


def absolute_indirect_coefs(fit: Any) -> dict[str, float]:
    """Recover β_indirect for each outcome from stacked parameterization."""
    b_ref = float(fit.params[FOCAL])
    out = {REF_OUTCOME: b_ref}
    for outcome, _ in OUTCOME_COLS:
        if outcome == REF_OUTCOME:
            continue
        diff_name = _param_containing(fit.params.index, FOCAL, outcome)
        out[outcome] = b_ref + float(fit.params[diff_name])
    return out


def wald_row(
    *,
    contrast: str,
    hypothesis: str,
    result: Any,
) -> dict[str, Any]:
    # statsmodels Wald returns F or chi2 depending on version / cov
    stat = float(np.asarray(result.statistic).reshape(-1)[0])
    pvalue = float(np.asarray(result.pvalue).reshape(-1)[0])
    df_num = getattr(result, "df_num", None)
    df_denom = getattr(result, "df_denom", None)
    if df_num is None and hasattr(result, "df_constraint"):
        df_num = result.df_constraint
    df_num_f = float(df_num) if df_num is not None else np.nan
    df_denom_f = float(df_denom) if df_denom is not None else np.nan
    if np.isfinite(df_num_f) and np.isfinite(df_denom_f):
        df_display = f"F({int(df_num_f)}, {int(df_denom_f)})"
    elif np.isfinite(df_num_f):
        df_display = f"df={int(df_num_f)}"
    else:
        df_display = ""
    return {
        "contrast": contrast,
        "hypothesis": hypothesis,
        "stat_type": "F",
        "stat": stat,
        "df_num": df_num_f,
        "df_denom": df_denom_f,
        "df_display": df_display,
        "pvalue": pvalue,
        "stars": _stars(pvalue),
    }


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _wald_r(fit: Any, rows: list[dict[str, float]]) -> Any:
    """Wald test via restriction matrix (avoids fragile param-name formulas)."""
    names = list(fit.params.index)
    pos = {n: i for i, n in enumerate(names)}
    r = np.zeros((len(rows), len(names)), dtype=float)
    for i, coeffs in enumerate(rows):
        for term, w in coeffs.items():
            r[i, pos[term]] = w
    return fit.wald_test(r, use_f=True, scalar=True)


def run_asymmetry_tests(fit: Any) -> pd.DataFrame:
    """Joint and pairwise Wald tests on Indirect/Mixed coefficients."""
    likes_diff = _param_containing(fit.params.index, FOCAL, "likes")
    comments_diff = _param_containing(fit.params.index, FOCAL, "comments")

    rows: list[dict[str, Any]] = []

    # Joint: β_L = β_R and β_C = β_R  ⇒  β_L = β_C = β_R
    joint = _wald_r(
        fit,
        [{likes_diff: 1.0}, {comments_diff: 1.0}],
    )
    rows.append(
        wald_row(
            contrast="joint_equality",
            hypothesis="beta_likes = beta_comments = beta_reposts",
            result=joint,
        )
    )

    # Pairwise differences vs reposts (= interaction = 0)
    for name, term, hyp in (
        ("likes_vs_reposts", likes_diff, "beta_likes - beta_reposts = 0"),
        ("comments_vs_reposts", comments_diff, "beta_comments - beta_reposts = 0"),
    ):
        res = _wald_r(fit, [{term: 1.0}])
        rows.append(wald_row(contrast=name, hypothesis=hyp, result=res))

    # likes vs comments: (β_L − β_R) − (β_C − β_R) = 0
    lc = _wald_r(fit, [{likes_diff: 1.0, comments_diff: -1.0}])
    rows.append(
        wald_row(
            contrast="likes_vs_comments",
            hypothesis="beta_likes - beta_comments = 0",
            result=lc,
        )
    )
    return pd.DataFrame(rows)


def component_coefs_separate(df: pd.DataFrame) -> pd.DataFrame:
    """Equation-by-equation OLS (same as script 06 components) for cross-check."""
    controls = " + ".join(PERIPHERAL_SAMPLE_CONTROLS)
    topic = f"C(topic_id, Treatment(reference='{REF_TOPIC}'))"
    rows: list[dict[str, Any]] = []
    for outcome, y_col in OUTCOME_COLS:
        formula = f"{y_col} ~ {FOCAL} + t2 + {controls} + {topic}"
        fit, n = fit_ols_cluster(df, formula)
        rows.append(
            {
                "source": "separate_ols",
                "outcome": outcome,
                "y_col": y_col,
                "term": FOCAL,
                "coef": float(fit.params[FOCAL]),
                "se": float(fit.bse[FOCAL]),
                "pvalue": float(fit.pvalues[FOCAL]),
                "n_posts": n,
                "stars": _stars(float(fit.pvalues[FOCAL])),
            }
        )
    return pd.DataFrame(rows)


def _t_linear(fit: Any, weights: dict[str, float]) -> tuple[float, float, float]:
    names = list(fit.params.index)
    pos = {n: i for i, n in enumerate(names)}
    r = np.zeros((1, len(names)), dtype=float)
    for term, w in weights.items():
        r[0, pos[term]] = w
    res = fit.t_test(r)
    coef = float(np.asarray(res.effect).reshape(-1)[0])
    se = float(np.asarray(res.sd).reshape(-1)[0])
    p = float(np.asarray(res.pvalue).reshape(-1)[0])
    return coef, se, p


def component_coefs_stacked(fit: Any, *, n_posts: int) -> pd.DataFrame:
    abs_b = absolute_indirect_coefs(fit)
    rows: list[dict[str, Any]] = []
    for outcome, y_col in OUTCOME_COLS:
        if outcome == REF_OUTCOME:
            coef, se, p = _t_linear(fit, {FOCAL: 1.0})
        else:
            diff = _param_containing(fit.params.index, FOCAL, outcome)
            coef, se, p = _t_linear(fit, {FOCAL: 1.0, diff: 1.0})
            if not np.isclose(coef, abs_b[outcome]):
                msg = f"stacked absolute coef mismatch for {outcome}"
                raise AssertionError(msg)
        rows.append(
            {
                "source": "stacked_ols",
                "outcome": outcome,
                "y_col": y_col,
                "term": FOCAL,
                "coef": coef,
                "se": se,
                "pvalue": p,
                "n_posts": n_posts,
                "stars": _stars(p),
            }
        )
    return pd.DataFrame(rows)


def prepare_h1_core(df: pd.DataFrame) -> pd.DataFrame:
    sample = apply_sample_mask(df, "h1")
    y_cols = [c for _, c in OUTCOME_COLS]
    extra = ["topic_id", "mid", *y_cols]
    # prepare on primary y then reattach other outcomes by index
    base = prepare_model_frame(
        sample,
        y_col="log_likes",
        x_cols=[FOCAL, "t2"],
        control_cols=PERIPHERAL_SAMPLE_CONTROLS,
        extra_cols=extra,
        categorical_cols=["topic_id"],
    )
    # Drop rows missing any component DV (common complete cases)
    for _, c in OUTCOME_COLS:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    base = base.dropna(subset=[c for _, c in OUTCOME_COLS])
    return base


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_MODEL_DATA_FINAL)
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str, "topic_id": str})
    core = prepare_h1_core(df)
    n_posts = len(core)
    n_accounts = int(core["account_id"].nunique())

    separate = component_coefs_separate(core)
    stacked_df = stack_engagement_outcomes(core)
    formula = stacked_formula()

    diag = diagnose_model_frame(
        stacked_df,
        formula,
        fitter="ols",
        y_col="y",
        x_cols=[FOCAL, "t2"],
        control_cols=PERIPHERAL_SAMPLE_CONTROLS,
    )
    if diag["zero_variance_cols"]:
        raise ZeroVarianceDiagnosticError(str(diag["zero_variance_cols"]))

    fit, n_stacked = fit_ols_cluster(stacked_df, formula)
    assert n_stacked == n_posts * 3

    stacked_coefs = component_coefs_stacked(fit, n_posts=n_posts)
    tests = run_asymmetry_tests(fit)

    # Cross-check: separate vs stacked point estimates should match OLS
    merged = separate.merge(
        stacked_coefs,
        on=["outcome", "term"],
        suffixes=("_sep", "_stk"),
    )
    max_abs_diff = float((merged["coef_sep"] - merged["coef_stk"]).abs().max())

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_PUBLIC.mkdir(parents=True, exist_ok=True)

    coefs = pd.concat([separate, stacked_coefs], ignore_index=True)
    coef_path = OUT_TABLES / "table_engagement_asymmetry_coefs.csv"
    test_path = OUT_TABLES / "table_engagement_asymmetry_wald.csv"
    public_path = OUT_PUBLIC / "table_engagement_asymmetry_wald.csv"
    coefs.to_csv(coef_path, index=False)
    tests.to_csv(test_path, index=False)
    tests.to_csv(public_path, index=False)

    summary_txt = format_diagnostic_header(diag, formula=formula)
    summary_txt += fit.summary().as_text()
    summary_txt += "\n\n=== Wald tests (Indirect/Mixed across outcomes) ===\n"
    summary_txt += tests.to_string(index=False)
    summary_txt += "\n\n=== Separate vs stacked |Δcoef| max ===\n"
    summary_txt += f"{max_abs_diff:.6g}\n"
    (OUT_MODELS / "h1_engagement_asymmetry_stacked.txt").write_text(summary_txt, encoding="utf-8")

    report = {
        "built_at": utc_now_iso(),
        "method": (
            "Stacked OLS with outcome×predictors; account-clustered SEs; "
            "reference outcome = reposts. Formal Wald tests of coefficient "
            "equality across likes / comments / reposts."
        ),
        "sample": "H1 core (disaster-impact × peripheral)",
        "n_posts": n_posts,
        "n_accounts": n_accounts,
        "n_stacked_rows": n_stacked,
        "formula": formula,
        "max_abs_coef_diff_separate_vs_stacked": max_abs_diff,
        "absolute_indirect_coefs": absolute_indirect_coefs(fit),
        "wald_tests": tests.to_dict(orient="records"),
        "paths": {
            "coefs": str(coef_path),
            "wald": str(test_path),
            "public_wald": str(public_path),
            "summary": str(OUT_MODELS / "h1_engagement_asymmetry_stacked.txt"),
        },
    }
    write_json_report(OUT_REPORTS / "engagement_asymmetry_report.json", report)

    print(f"n_posts={n_posts}, n_accounts={n_accounts}, stacked={n_stacked}")
    print(tests.to_string(index=False))
    print(f"max |Δcoef| separate vs stacked: {max_abs_diff:.6g}")
    print(f"wrote {test_path}")
    print(f"wrote {public_path}")


if __name__ == "__main__":
    main()
