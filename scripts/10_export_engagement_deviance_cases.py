"""Export positive/negative residual deviance cases (Indirect-Mixed) for qualitative work.

Fits the structural expectation OLS on the H1 core sample, screens viral/review
outliers, then selects top/bottom residual Indirect-Mixed posts. Negative
deviance is restricted to raw engagement > 0 (Plan A). A robustness spec that
adds relative_window_day is used only to check residual-rank stability.
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
    PATH_ANALYSIS_TOPICS,
    PATH_MODEL_DATA_FINAL,
    PROJECT_ROOT,
    ModelSpec,
    ZeroVarianceDiagnosticError,
    apply_sample_mask,
    build_formula,
    diagnose_model_frame,
    fit_ols_cluster,
    format_diagnostic_header,
    prepare_model_frame,
    spec_by_spec_id,
    utc_now_iso,
    write_json_report,
)

OUT_CASES = PROJECT_ROOT / "outputs" / "cases"
OUT_MODELS = PROJECT_ROOT / "outputs" / "models"

CARRY_COLS = (
    "mid",
    "account_id",
    "text_hash",
    "indirect_clean",
    "label_expression_clean",
    "label_narrative_clean",
    "label_emotion_clean",
    "t2",
    "relative_window_day",
    "log_followers",
    "post_hour_cst",
    "hashtag_count",
    "text_length",
    "topic_id",
    "log_engagement",
    "likes_count",
    "comments_count",
    "reposts_count",
    "high_clarity",
    "needs_manual_review",
)


def raw_engagement_series(df: pd.DataFrame) -> pd.Series:
    likes = pd.to_numeric(df.get("likes_count"), errors="coerce").fillna(0)
    comments = pd.to_numeric(df.get("comments_count"), errors="coerce").fillna(0)
    reposts = pd.to_numeric(df.get("reposts_count"), errors="coerce").fillna(0)
    return likes + comments + reposts


def fit_expectation_with_residuals(
    df: pd.DataFrame,
    spec: ModelSpec,
) -> tuple[pd.DataFrame, Any, str]:
    """Fit OLS on the H1 core sample; return estimation frame with fitted/resid."""
    eligible = apply_sample_mask(df, spec.sample_key).copy().reset_index(drop=True)
    cat_cols = [c for c, _ in spec.categorical_cols]
    extra_cols: list[str] = list(spec.extra_controls)
    if spec.use_topic_fe:
        eligible["topic_id"] = eligible["topic_id"].astype("Int64").astype(str)
        cat_cols = [*cat_cols, "topic_id"]
    y = spec.y_col
    x_cols = list(spec.x_cols)
    # Only model columns go through prepare_model_frame (avoids coercing mid/labels).
    frame = prepare_model_frame(
        eligible,
        y_col=y,
        x_cols=x_cols,
        control_cols=spec.control_cols,
        extra_cols=extra_cols or None,
        categorical_cols=cat_cols or None,
    )
    formula = build_formula(spec, y_col=y)
    diagnostics = diagnose_model_frame(
        frame,
        formula,
        fitter=spec.fitter,
        y_col=y,
        x_cols=x_cols,
        control_cols=spec.control_cols,
    )
    if diagnostics["zero_variance_cols"]:
        msg = (
            f"Zero-variance predictors in {spec.model_id} ({spec.default_spec_id}): "
            f"{diagnostics['zero_variance_cols']}. formula={formula!r}"
        )
        raise ZeroVarianceDiagnosticError(msg)

    fit, _n = fit_ols_cluster(frame, formula)
    out = frame.copy()
    out["predicted_log_engagement"] = np.asarray(fit.fittedvalues, dtype=float)
    out["residual_engagement"] = np.asarray(fit.resid, dtype=float)
    # Re-attach export fields by preserved index
    for c in CARRY_COLS:
        if c in eligible.columns and c not in out.columns:
            out[c] = eligible.loc[out.index, c].to_numpy()
        elif c in eligible.columns and c == "mid":
            out[c] = eligible.loc[out.index, c].astype(str).to_numpy()
    if "mid" not in out.columns and "mid" in eligible.columns:
        out["mid"] = eligible.loc[out.index, "mid"].astype(str).to_numpy()
    out["raw_engagement"] = raw_engagement_series(out)
    summary = format_diagnostic_header(diagnostics, formula=formula) + str(fit.summary())
    return out, fit, summary


def _as_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    mapped = s.map(
        lambda v: str(v).strip().lower() in {"1", "true", "yes", "t"}
        if pd.notna(v)
        else False
    )
    return mapped.astype(bool)


def apply_preselect_screens(
    residuals: pd.DataFrame,
    *,
    viral_percentile: float,
) -> pd.DataFrame:
    """Flag viral events and needs_manual_review (does not drop rows)."""
    out = residuals.copy()
    raw = pd.to_numeric(out["raw_engagement"], errors="coerce")
    viral_cut = float(np.nanpercentile(raw.to_numpy(), viral_percentile))
    if "needs_manual_review" in out.columns:
        needs_flag = _as_bool_series(out["needs_manual_review"])
    else:
        needs_flag = pd.Series(False, index=out.index)

    reasons: list[str] = []
    passes: list[bool] = []
    zero_eng: list[bool] = []
    for i in range(len(out)):
        reason = ""
        is_zero = float(raw.iloc[i]) <= 0
        zero_eng.append(is_zero)
        if bool(needs_flag.iloc[i]):
            reason = "screen_needs_review"
        elif float(raw.iloc[i]) > viral_cut:
            reason = "screen_viral_event"
        # zero_engagement flagged separately; excluded only from negative selection
        reasons.append(reason)
        passes.append(reason == "")
    out["screen_reason"] = reasons
    out["screen_pass"] = passes
    out["zero_engagement"] = zero_eng
    out.attrs["viral_cut"] = viral_cut
    out.attrs["n_zero_engagement"] = int(sum(zero_eng))
    return out


def select_deviance_cases(
    screened: pd.DataFrame,
    *,
    k: int,
    max_per_account: int,
    residual_col: str = "residual_engagement",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select top/bottom residual Indirect-Mixed cases after preselect screens.

    Negative deviance (Plan A): only among posts with raw engagement > 0, so
    zero-interaction posts are not mistaken for true under-performance.
    """
    base = screened.loc[
        (screened["indirect_clean"] == 1) & (screened["screen_pass"])
    ].copy()
    raw = pd.to_numeric(base.get("raw_engagement"), errors="coerce").fillna(0)

    def take_side(
        pool: pd.DataFrame,
        *,
        ascending: bool,
        deviance_type: str,
    ) -> pd.DataFrame:
        ordered = pool.sort_values(residual_col, ascending=ascending, kind="mergesort")
        kept: list[pd.Series] = []
        per_account: dict[str, int] = {}
        for _, row in ordered.iterrows():
            aid = str(row["account_id"])
            if per_account.get(aid, 0) >= max_per_account:
                continue
            per_account[aid] = per_account.get(aid, 0) + 1
            kept.append(row)
            if len(kept) >= k:
                break
        if not kept:
            empty = ordered.iloc[0:0].copy()
            empty["deviance_type"] = pd.Series(dtype=str)
            empty["residual_rank"] = pd.Series(dtype=int)
            return empty
        out = pd.DataFrame(kept)
        out["deviance_type"] = deviance_type
        out["residual_rank"] = np.arange(1, len(out) + 1)
        return out

    positive = take_side(base, ascending=False, deviance_type="positive")
    # Plan A: true under-performers require some observed engagement.
    neg_pool = base.loc[raw > 0].copy()
    negative = take_side(neg_pool, ascending=True, deviance_type="negative")
    return positive, negative


def mark_account_cap_in_full(
    screened: pd.DataFrame,
    positive: pd.DataFrame,
    negative: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate full residual table with account-cap exclusions among Indirect-Mixed."""
    out = screened.copy()
    selected = set()
    if "mid" in positive.columns:
        selected.update(positive["mid"].astype(str))
    if "mid" in negative.columns:
        selected.update(negative["mid"].astype(str))

    # Rows that pass preselect, are Indirect-Mixed, extreme-ish, but not selected
    # due to account cap get screen_account_cap only if same account already selected
    # on that side. Simpler: leave screen_reason as-is for non-selected; document in report.
    if "mid" in out.columns:
        out["selected_deviance"] = out["mid"].astype(str).isin(selected)
    else:
        out["selected_deviance"] = False
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def residual_rank_stability(
    main_resid: pd.DataFrame,
    robust_resid: pd.DataFrame,
    *,
    k: int,
    max_per_account: int,
    viral_percentile: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare main vs +relative_window_day residual rankings on Indirect-Mixed."""
    m = main_resid[["mid", "residual_engagement", "indirect_clean"]].copy()
    r = robust_resid[["mid", "residual_engagement"]].rename(
        columns={"residual_engagement": "residual_engagement_robust"}
    )
    merged = m.merge(r, on="mid", how="inner")
    im = merged.loc[merged["indirect_clean"] == 1].copy()
    if im.empty:
        empty = pd.DataFrame(
            columns=[
                "side",
                "spearman_rho",
                "jaccard_top_k",
                "n_overlap",
                "n_main",
                "n_robust",
            ]
        )
        return empty, {"n_indirect_mixed_overlap": 0, "spearman_rho": np.nan}

    rho = float(im["residual_engagement"].corr(im["residual_engagement_robust"], method="spearman"))

    # Re-run selection on each residual series using the same screening from main table
    main_screened = apply_preselect_screens(main_resid, viral_percentile=viral_percentile)
    # Align robust residuals onto main screened frame for fair compare
    robust_on_main = main_screened.merge(
        r,
        on="mid",
        how="left",
    )
    pos_m, neg_m = select_deviance_cases(
        main_screened, k=k, max_per_account=max_per_account, residual_col="residual_engagement"
    )
    # For robust side, temporarily use robust residual as selection key
    robust_sel = robust_on_main.copy()
    robust_sel["residual_engagement"] = robust_sel["residual_engagement_robust"]
    pos_r, neg_r = select_deviance_cases(
        robust_sel, k=k, max_per_account=max_per_account, residual_col="residual_engagement"
    )

    rows = []
    for side, sm, sr in (
        ("positive", pos_m, pos_r),
        ("negative", neg_m, neg_r),
    ):
        set_m = set(sm["mid"].astype(str)) if len(sm) else set()
        set_r = set(sr["mid"].astype(str)) if len(sr) else set()
        rows.append(
            {
                "side": side,
                "spearman_rho": rho,
                "jaccard_top_k": jaccard(set_m, set_r),
                "n_overlap": len(set_m & set_r),
                "n_main": len(set_m),
                "n_robust": len(set_r),
            }
        )
    summary = {
        "n_indirect_mixed_overlap": int(len(im)),
        "spearman_rho": rho,
        "positive_jaccard": rows[0]["jaccard_top_k"],
        "negative_jaccard": rows[1]["jaccard_top_k"],
    }
    return pd.DataFrame(rows), summary


def attach_robust_ranks(
    cases: pd.DataFrame,
    robust_resid: pd.DataFrame,
) -> pd.DataFrame:
    if cases.empty:
        out = cases.copy()
        out["residual_engagement_robust"] = pd.Series(dtype=float)
        out["residual_rank_robust"] = pd.Series(dtype=float)
        return out
    r = robust_resid[["mid", "residual_engagement"]].rename(
        columns={"residual_engagement": "residual_engagement_robust"}
    )
    # Rank among Indirect-Mixed in robust frame (1 = highest residual)
    rob_im = robust_resid.loc[
        robust_resid["indirect_clean"] == 1, ["mid", "residual_engagement"]
    ].copy()
    rob_im["residual_rank_robust"] = (
        rob_im["residual_engagement"].rank(ascending=False, method="min").astype(int)
    )
    r = r.merge(rob_im[["mid", "residual_rank_robust"]], on="mid", how="left")
    return cases.merge(r, on="mid", how="left")


def merge_text(cases: pd.DataFrame, text_source: Path | None) -> pd.DataFrame:
    if text_source is None or not text_source.is_file() or cases.empty:
        out = cases.copy()
        if "analysis_text_final" not in out.columns:
            out["analysis_text_final"] = pd.NA
        return out
    text_df = pd.read_csv(text_source, dtype={"mid": str}, low_memory=False)
    cols = [c for c in ("mid", "analysis_text_final", "text", "text_hash") if c in text_df.columns]
    if "mid" not in cols:
        return cases
    slim = text_df[cols].drop_duplicates(subset=["mid"])
    if "analysis_text_final" not in slim.columns and "text" in slim.columns:
        slim = slim.rename(columns={"text": "analysis_text_final"})
    keep = [c for c in ("mid", "analysis_text_final", "text_hash") if c in slim.columns]
    return cases.merge(slim[keep], on="mid", how="left", suffixes=("", "_txt"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_MODEL_DATA_FINAL)
    p.add_argument("--text-source", type=Path, default=PATH_ANALYSIS_TOPICS)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--viral-percentile", type=float, default=99.5)
    p.add_argument("--max-per-account", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=OUT_CASES)
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(
            f"input not found: {inp}\nRun: uv run python scripts/03_prepare_model_data.py"
        )

    df = pd.read_csv(inp, dtype={"mid": str})
    if "verified" in df.columns:
        df["verified"] = df["verified"].fillna(0).astype(int)

    main_spec = spec_by_spec_id("expectation_main")
    robust_spec = spec_by_spec_id("expectation_with_relative_day")

    main_frame, _main_fit, main_summary = fit_expectation_with_residuals(df, main_spec)
    robust_frame, _rob_fit, rob_summary = fit_expectation_with_residuals(df, robust_spec)

    screened = apply_preselect_screens(main_frame, viral_percentile=args.viral_percentile)
    viral_cut = float(screened.attrs.get("viral_cut", np.nan))

    positive, negative = select_deviance_cases(
        screened,
        k=args.k,
        max_per_account=args.max_per_account,
    )
    full = mark_account_cap_in_full(screened, positive, negative)

    # Merge robust residual onto full table
    rob_slim = robust_frame[["mid", "residual_engagement", "predicted_log_engagement"]].rename(
        columns={
            "residual_engagement": "residual_engagement_robust",
            "predicted_log_engagement": "predicted_log_engagement_robust",
        }
    )
    full = full.merge(rob_slim, on="mid", how="left")

    positive = attach_robust_ranks(positive, robust_frame)
    negative = attach_robust_ranks(negative, robust_frame)

    text_path = args.text_source.expanduser().resolve() if args.text_source else None
    positive = merge_text(positive, text_path)
    negative = merge_text(negative, text_path)

    stab_df, stab_summary = residual_rank_stability(
        main_frame,
        robust_frame,
        k=args.k,
        max_per_account=args.max_per_account,
        viral_percentile=args.viral_percentile,
    )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)

    full.to_csv(out_dir / "engagement_residuals_core.csv", index=False, encoding="utf-8-sig")
    positive.to_csv(
        out_dir / "positive_deviance_indirect_mixed.csv", index=False, encoding="utf-8-sig"
    )
    negative.to_csv(
        out_dir / "negative_deviance_indirect_mixed.csv", index=False, encoding="utf-8-sig"
    )
    stab_df.to_csv(out_dir / "residual_rank_stability.csv", index=False, encoding="utf-8-sig")

    (OUT_MODELS / "expectation_engagement_expression.txt").write_text(
        main_summary, encoding="utf-8"
    )
    (OUT_MODELS / "expectation_engagement_expression_with_relative_day.txt").write_text(
        rob_summary, encoding="utf-8"
    )

    n_screen_fail = int((~full["screen_pass"]).sum())
    n_zero = int(full["zero_engagement"].sum()) if "zero_engagement" in full.columns else 0
    report: dict[str, Any] = {
        "built_at": utc_now_iso(),
        "input": str(inp),
        "text_source": str(text_path) if text_path else None,
        "formula_main": build_formula(main_spec),
        "formula_robust": build_formula(robust_spec),
        "n_estimation_main": int(len(main_frame)),
        "n_estimation_robust": int(len(robust_frame)),
        "k": args.k,
        "viral_percentile": args.viral_percentile,
        "viral_cut_raw_engagement": viral_cut,
        "max_per_account": args.max_per_account,
        "n_screen_fail_preselect": n_screen_fail,
        "n_zero_engagement_core": n_zero,
        "negative_requires_engagement_gt_0": True,
        "n_positive_selected": int(len(positive)),
        "n_negative_selected": int(len(negative)),
        "rank_stability": stab_summary,
        "note": (
            "Case selection uses main-spec residuals after viral/review screens "
            "and per-account cap. Negative deviance restricted to raw engagement > 0 "
            "(Plan A: exclude zero likes/comments/reposts). "
            "relative_window_day spec is for rank stability only."
        ),
    }
    write_json_report(out_dir / "case_selection_report.json", report)

    print(
        f"deviance cases -> {out_dir} "
        f"(pos={len(positive)}, neg={len(negative)}, "
        f"spearman={stab_summary.get('spearman_rho')})"
    )


if __name__ == "__main__":
    main()
