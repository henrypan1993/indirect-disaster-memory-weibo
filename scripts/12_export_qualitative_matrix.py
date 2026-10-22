"""Export the qualitative sampling matrix (A1/A2/B/C/contrasts).

A1/A2 do not use engagement or residuals. A2 Indirect is a subset of A1 core
Indirect with 6 Trauma + 2 Memory per period. B/C use component residuals with
period-specific scores/percentiles. One row per mid; selection_role is multi-value.
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
    QUALITATIVE_EXCLUSION_REASONS,
    SEMANTIC_ANALYSIS_IDS,
    TRAUMA_NARRATIVES_CLEAN,
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
OUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"

ROLE_A1 = "A1_contextual_transformation"
ROLE_A2_IND = "A2_indirect_form_contrast"
ROLE_A2_DIR = "A2_direct_form_contrast"
ROLE_B = "B_bounded_resonance"
ROLE_C = "C_portability_deviant"
ROLE_NEG = "low_engagement_contrast"
ROLE_UNCLEAR = "sensitivity_unclear"

NARR_TRAUMA = "Trauma-Help-Loss"
NARR_MEMORY = "Memory-Reactivation"
NARR_WEATHER = "Ordinary-Weather-Info"
NARR_UNCLEAR = "Other-Unclear"

A1_QUOTAS: dict[str, int] = {NARR_TRAUMA: 7, NARR_MEMORY: 4, NARR_WEATHER: 4}
A2_QUOTAS: dict[str, int] = {NARR_TRAUMA: 6, NARR_MEMORY: 2}

A_ELIGIBILITY_COLS = frozenset(
    {
        "mid",
        "account_id",
        "t2",
        "indirect_clean",
        "label_narrative_clean",
        "label_expression_clean",
        "needs_manual_review",
        "high_clarity",
        "analysis_text_final",
        "text_hash",
        "text_length",
        "disaster_impact_related",
        "sample_h1_eligible",
        "sample_h1_broader_eligible",
    }
)

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
    "log_likes",
    "log_comments",
    "log_reposts",
    "likes_count",
    "comments_count",
    "reposts_count",
    "high_clarity",
    "needs_manual_review",
    "disaster_impact_related",
    "sample_h1_eligible",
    "sample_h1_broader_eligible",
    "analysis_text_final",
)

SAMPLE_A_NOTE = (
    "Sample A eligibility uses text analyzability, coding certainty, account "
    "cap, narrative stratum, period, and expression form only. Engagement "
    "counts and residuals are not used to include or exclude A cases. "
    "Other-Unclear is not a principal matched stratum because its residual "
    "category status does not provide a substantively coherent discourse type."
)


def _as_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    mapped = s.map(
        lambda v: str(v).strip().lower() in {"1", "true", "yes", "t"} if pd.notna(v) else False
    )
    return mapped.astype(bool)


def raw_engagement_series(df: pd.DataFrame) -> pd.Series:
    likes = pd.to_numeric(df.get("likes_count"), errors="coerce").fillna(0)
    comments = pd.to_numeric(df.get("comments_count"), errors="coerce").fillna(0)
    reposts = pd.to_numeric(df.get("reposts_count"), errors="coerce").fillna(0)
    return likes + comments + reposts


def year_label_series(t2: pd.Series) -> pd.Series:
    t = pd.to_numeric(t2, errors="coerce").fillna(0).astype(int)
    return t.map({0: "2021_T1", 1: "2025_T2"}).astype("string")


def analyzable_text_mask(df: pd.DataFrame) -> pd.Series:
    if "analysis_text_final" in df.columns:
        s = df["analysis_text_final"].astype("string").str.strip()
        ok = s.notna() & (s != "") & (s != "<NA>")
        if bool(ok.any()):
            return ok
    if "text_length" in df.columns:
        return pd.to_numeric(df["text_length"], errors="coerce").fillna(0) > 0
    return pd.Series(True, index=df.index)


def a_eligibility_mask(df: pd.DataFrame) -> pd.Series:
    """Eligibility for Sample A. Must not consult engagement or residuals."""
    cols = [c for c in A_ELIGIBILITY_COLS if c in df.columns]
    view = df.loc[:, cols]
    needs = (
        _as_bool_series(view["needs_manual_review"])
        if "needs_manual_review" in view.columns
        else pd.Series(False, index=view.index)
    )
    broader = (
        pd.to_numeric(view["sample_h1_broader_eligible"], errors="coerce").fillna(0).eq(1)
        if "sample_h1_broader_eligible" in view.columns
        else pd.Series(True, index=view.index)
    )
    return analyzable_text_mask(view) & (~needs) & broader


def take_rows(
    pool: pd.DataFrame,
    *,
    n: int,
    seed: int,
    max_per_account: int = 1,
    rank_col: str | None = None,
    ascending: bool = False,
) -> pd.DataFrame:
    if pool.empty or n <= 0:
        return pool.iloc[0:0].copy()
    work = pool.copy()
    if rank_col and rank_col in work.columns:
        ordered = work.sort_values(rank_col, ascending=ascending, kind="mergesort")
    else:
        rng = np.random.default_rng(seed)
        ordered = work.iloc[rng.permutation(len(work))]
    kept: list[pd.Series] = []
    counts: dict[str, int] = {}
    for _, row in ordered.iterrows():
        aid = str(row.get("account_id", ""))
        if counts.get(aid, 0) >= max_per_account:
            continue
        counts[aid] = counts.get(aid, 0) + 1
        kept.append(row)
        if len(kept) >= n:
            break
    if not kept:
        return ordered.iloc[0:0].copy()
    return pd.DataFrame(kept)


def _mark_selected(candidates: pd.DataFrame, n_selected: int) -> pd.DataFrame:
    out = candidates.copy().reset_index(drop=True)
    n = min(max(n_selected, 0), len(out))
    out["eligible"] = True
    out["selected"] = [i < n for i in range(len(out))]
    out["exclusion_reason"] = ""
    return out


def draw_a1_stratum(
    pool: pd.DataFrame,
    *,
    t2: int,
    narrative: str,
    quota: int,
    seed: int,
    oversample: int = 2,
) -> pd.DataFrame:
    sub = pool.loc[
        (pd.to_numeric(pool["t2"], errors="coerce") == t2)
        & (pool["label_narrative_clean"].astype("string") == narrative)
        & (pd.to_numeric(pool["indirect_clean"], errors="coerce") == 1)
    ]
    n_cand = quota * oversample
    cand = take_rows(sub, n=n_cand, seed=seed + t2 * 100 + quota)
    out = _mark_selected(cand, quota)
    out["stratum"] = f"A1|t2={t2}|{narrative}"
    out["selection_role"] = ROLE_A1
    return out


def draw_a1(broader_indirect: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    eligible = broader_indirect.loc[a_eligibility_mask(broader_indirect)].copy()
    parts = [
        draw_a1_stratum(
            eligible,
            t2=t2,
            narrative=narr,
            quota=q,
            seed=seed,
        )
        for t2 in (0, 1)
        for narr, q in A1_QUOTAS.items()
    ]
    return pd.concat(parts, ignore_index=True) if parts else eligible.iloc[0:0].copy()


def draw_a2_indirect_from_a1(a1: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """A2 Indirect must be a subset of A1 selected core (Trauma/Memory) cases."""
    base = a1.loc[
        (a1["selected"])
        & (a1["label_narrative_clean"].astype("string").isin(TRAUMA_NARRATIVES_CLEAN))
    ].copy()
    parts: list[pd.DataFrame] = []
    for t2 in (0, 1):
        period = base.loc[pd.to_numeric(base["t2"], errors="coerce") == t2]
        for narr, q in A2_QUOTAS.items():
            sub = period.loc[period["label_narrative_clean"].astype("string") == narr]
            taken = take_rows(sub, n=q, seed=seed + 500 + t2 * 10 + q)
            taken = _mark_selected(taken, q)
            taken["stratum"] = f"A2_indirect|t2={t2}|{narr}"
            taken["selection_role"] = ROLE_A2_IND
            parts.append(taken)
    return pd.concat(parts, ignore_index=True) if parts else base.iloc[0:0].copy()


def draw_a2_direct(core: pd.DataFrame, *, seed: int, oversample: int = 2) -> pd.DataFrame:
    eligible = core.loc[
        a_eligibility_mask(core)
        & (pd.to_numeric(core["indirect_clean"], errors="coerce") == 0)
        & (core["label_narrative_clean"].astype("string").isin(TRAUMA_NARRATIVES_CLEAN))
    ].copy()
    parts: list[pd.DataFrame] = []
    for t2 in (0, 1):
        period = eligible.loc[pd.to_numeric(eligible["t2"], errors="coerce") == t2]
        for narr, q in A2_QUOTAS.items():
            sub = period.loc[period["label_narrative_clean"].astype("string") == narr]
            cand = take_rows(sub, n=q * oversample, seed=seed + 800 + t2 * 20 + q)
            taken = _mark_selected(cand, q)
            taken["stratum"] = f"A2_direct|t2={t2}|{narr}"
            taken["selection_role"] = ROLE_A2_DIR
            parts.append(taken)
    return pd.concat(parts, ignore_index=True) if parts else eligible.iloc[0:0].copy()


def period_zscore(values: pd.Series, period: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    x_all = pd.to_numeric(values, errors="coerce")
    p_all = pd.to_numeric(period, errors="coerce")
    for p in (0, 1):
        m = p_all == p
        x = x_all.loc[m]
        if x.empty:
            continue
        sd = float(x.std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            out.loc[m] = 0.0
        else:
            out.loc[m] = (x - float(x.mean())) / sd
    return out


def bounded_resonance_eligible(df: pd.DataFrame) -> pd.Series:
    r_l = pd.to_numeric(df["likes_resid"], errors="coerce")
    r_c = pd.to_numeric(df["comments_resid"], errors="coerce")
    r_r = pd.to_numeric(df["reposts_resid"], errors="coerce")
    indirect = pd.to_numeric(df["indirect_clean"], errors="coerce") == 1
    return indirect & (r_l > 0) & (r_c > 0) & (r_r <= 0)


def profile_score_b(df: pd.DataFrame) -> pd.Series:
    t2 = df["t2"]
    z_l = period_zscore(df["likes_resid"], t2)
    z_c = period_zscore(df["comments_resid"], t2)
    z_r = period_zscore(df["reposts_resid"], t2)
    return z_l + z_c - z_r


def period_percentile_threshold(values: pd.Series, period: pd.Series, q: float) -> pd.Series:
    """Boolean: value strictly above (q>0.5) or at/below (q<=0.5) period percentile."""
    x = pd.to_numeric(values, errors="coerce")
    p = pd.to_numeric(period, errors="coerce")
    out = pd.Series(False, index=values.index)
    for per in (0, 1):
        m = p == per
        if int(m.sum()) == 0:
            continue
        cut = float(np.nanpercentile(x.loc[m].to_numpy(dtype=float), q * 100))
        if q >= 0.5:
            out.loc[m] = x.loc[m] > cut
        else:
            out.loc[m] = x.loc[m] <= cut
    return out


def select_b(
    scored: pd.DataFrame,
    *,
    n_per_period: int = 10,
    oversample: int = 2,
    seed: int = 12,
) -> pd.DataFrame:
    pool = scored.loc[bounded_resonance_eligible(scored)].copy()
    if "screen_pass_b" in pool.columns:
        pool = pool.loc[pool["screen_pass_b"]].copy()
    pool["score_b"] = profile_score_b(pool)
    parts: list[pd.DataFrame] = []
    for t2 in (0, 1):
        sub = pool.loc[pd.to_numeric(pool["t2"], errors="coerce") == t2]
        cand = take_rows(
            sub,
            n=n_per_period * oversample,
            seed=seed + t2,
            rank_col="score_b",
            ascending=False,
        )
        taken = _mark_selected(cand, n_per_period)
        taken["stratum"] = f"B|t2={t2}"
        taken["selection_role"] = ROLE_B
        parts.append(taken)
    return pd.concat(parts, ignore_index=True) if parts else pool.iloc[0:0].copy()


def select_c(
    scored: pd.DataFrame,
    *,
    n_per_period: int = 4,
    oversample: int = 2,
    min_reposts: int = 2,
    seed: int = 12,
) -> pd.DataFrame:
    work = scored.copy()
    indirect = pd.to_numeric(work["indirect_clean"], errors="coerce") == 1
    raw_r = pd.to_numeric(work.get("reposts_count"), errors="coerce").fillna(0)
    ind = work.loc[indirect]
    top = pd.Series(False, index=work.index)
    if not ind.empty:
        top.loc[ind.index] = period_percentile_threshold(ind["reposts_resid"], ind["t2"], 0.90)
    pool = work.loc[indirect & top & (raw_r >= min_reposts)].copy()
    if "screen_pass_c" in pool.columns:
        pool = pool.loc[pool["screen_pass_c"]].copy()
    parts: list[pd.DataFrame] = []
    for t2 in (0, 1):
        sub = pool.loc[pd.to_numeric(pool["t2"], errors="coerce") == t2]
        cand = take_rows(
            sub,
            n=n_per_period * oversample,
            seed=seed + 20 + t2,
            rank_col="reposts_resid",
            ascending=False,
        )
        taken = _mark_selected(cand, n_per_period)
        taken["stratum"] = f"C|t2={t2}"
        taken["selection_role"] = ROLE_C
        parts.append(taken)
    return pd.concat(parts, ignore_index=True) if parts else pool.iloc[0:0].copy()


def select_contrasts(
    scored: pd.DataFrame,
    *,
    n_per_period: int = 3,
    oversample: int = 2,
    seed: int = 12,
) -> pd.DataFrame:
    work = scored.copy()
    pred = pd.to_numeric(work["predicted_log_engagement"], errors="coerce")
    t2 = pd.to_numeric(work["t2"], errors="coerce")
    above_med = pd.Series(False, index=work.index)
    for per in (0, 1):
        m = t2 == per
        if int(m.sum()) == 0:
            continue
        med = float(pred.loc[m].median())
        above_med.loc[m] = pred.loc[m] >= med
    bottom = period_percentile_threshold(work["residual_engagement"], work["t2"], 0.10)
    pool = work.loc[above_med & bottom].copy()
    if "screen_pass_b" in pool.columns:
        pool = pool.loc[pool["screen_pass_b"]].copy()
    parts: list[pd.DataFrame] = []
    for per in (0, 1):
        sub = pool.loc[pd.to_numeric(pool["t2"], errors="coerce") == per]
        cand = take_rows(
            sub,
            n=n_per_period * oversample,
            seed=seed + 40 + per,
            rank_col="residual_engagement",
            ascending=True,
        )
        taken = _mark_selected(cand, n_per_period)
        taken["stratum"] = f"contrast|t2={per}"
        taken["selection_role"] = ROLE_NEG
        parts.append(taken)
    return pd.concat(parts, ignore_index=True) if parts else pool.iloc[0:0].copy()


def select_unclear(
    broader: pd.DataFrame,
    *,
    n: int = 3,
    oversample: int = 2,
    seed: int = 12,
) -> pd.DataFrame:
    pool = broader.loc[
        a_eligibility_mask(broader)
        & (pd.to_numeric(broader["indirect_clean"], errors="coerce") == 1)
        & (broader["label_narrative_clean"].astype("string") == NARR_UNCLEAR)
    ].copy()
    cand = take_rows(pool, n=n * oversample, seed=seed + 90)
    taken = _mark_selected(cand, n)
    taken["stratum"] = "sensitivity|Other-Unclear"
    taken["selection_role"] = ROLE_UNCLEAR
    return taken


def apply_bc_screens(df: pd.DataFrame, *, viral_percentile: float = 99.5) -> pd.DataFrame:
    out = df.copy()
    raw = raw_engagement_series(out)
    out["raw_engagement"] = raw
    reposts = pd.to_numeric(out.get("reposts_count"), errors="coerce").fillna(0)
    out["repost_share"] = np.where(raw > 0, reposts / raw, 0.0)
    arr = raw.to_numpy(dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        viral_cut = float("inf")
    else:
        viral_cut = float(np.nanpercentile(arr, viral_percentile))
    needs = (
        _as_bool_series(out["needs_manual_review"])
        if "needs_manual_review" in out.columns
        else pd.Series(False, index=out.index)
    )
    viral = raw > viral_cut
    out["viral_flag"] = viral
    out["screen_pass_b"] = (~needs) & (~viral)
    out["screen_pass_c"] = ~needs
    out.attrs["viral_cut"] = viral_cut
    return out


def enrich_unique_with_scores(
    unique: pd.DataFrame,
    scored: pd.DataFrame,
    b: pd.DataFrame,
) -> pd.DataFrame:
    """Attach component residuals / flags without duplicating mids."""
    out = unique.copy()
    if out.empty:
        return out
    score_cols = [
        c
        for c in (
            "likes_resid",
            "comments_resid",
            "reposts_resid",
            "residual_engagement",
            "predicted_log_engagement",
            "raw_engagement",
            "repost_share",
            "viral_flag",
        )
        if c in scored.columns
    ]
    if score_cols and "mid" in scored.columns:
        extra = scored.loc[:, ["mid", *score_cols]].drop_duplicates(subset=["mid"])
        out = out.drop(columns=[c for c in score_cols if c in out.columns], errors="ignore")
        out = out.merge(extra, on="mid", how="left")
    if b is not None and not b.empty and "score_b" in b.columns:
        b_scores = b.loc[:, ["mid", "score_b"]].drop_duplicates(subset=["mid"])
        if "score_b" in out.columns:
            out = out.drop(columns=["score_b"])
        out = out.merge(b_scores, on="mid", how="left")
    return out


def ordered_export(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    front = [
        c
        for c in (
            *CARRY_COLS,
            "year_label",
            "selection_role",
            "unique_case",
            "selected",
            "stratum",
            "eligible",
            "exclusion_reason",
            "likes_resid",
            "comments_resid",
            "reposts_resid",
            "residual_engagement",
            "predicted_log_engagement",
            "score_b",
            "repost_share",
            "raw_engagement",
            "viral_flag",
        )
        if c in df.columns
    ]
    rest = [c for c in df.columns if c not in front]
    return df.loc[:, [*front, *rest]]


def merge_unique_cases(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [f for f in frames if f is not None and not f.empty]
    if not nonempty:
        return pd.DataFrame(columns=["mid", "selection_role"])
    roles: dict[str, set[str]] = {}
    selected_mids: set[str] = set()
    rows_by_mid: dict[str, pd.Series] = {}
    for frame in nonempty:
        for _, row in frame.iterrows():
            mid = str(row["mid"])
            role = str(row.get("selection_role", ""))
            roles.setdefault(mid, set())
            if role:
                roles[mid].add(role)
            if bool(row.get("selected", False)):
                selected_mids.add(mid)
            if mid not in rows_by_mid:
                rows_by_mid[mid] = row
    recs = []
    for mid, row in rows_by_mid.items():
        rec = row.to_dict()
        rec["mid"] = mid
        rec["selection_role"] = ";".join(sorted(roles.get(mid, set())))
        rec["unique_case"] = True
        rec["selected"] = mid in selected_mids
        recs.append(rec)
    out = pd.DataFrame(recs)
    if "t2" in out.columns:
        out["year_label"] = year_label_series(out["t2"])
    return out


def build_selection_log(frames: list[pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        cols = [
            c
            for c in (
                "mid",
                "stratum",
                "eligible",
                "selected",
                "exclusion_reason",
                "selection_role",
                "account_id",
                "t2",
                "label_narrative_clean",
                "indirect_clean",
            )
            if c in frame.columns
        ]
        parts.append(frame[cols].copy())
    if not parts:
        return pd.DataFrame(columns=["mid", "stratum", "eligible", "selected", "exclusion_reason"])
    log = pd.concat(parts, ignore_index=True)
    log["exclusion_reason"] = log["exclusion_reason"].fillna("").astype(str)
    return log


def validate_exclusion_reason(reason: str) -> None:
    if reason in {"", None}:
        return
    if reason not in QUALITATIVE_EXCLUSION_REASONS:
        msg = f"disallowed exclusion_reason: {reason!r}"
        raise ValueError(msg)


def fit_y_with_residuals(
    df: pd.DataFrame,
    *,
    y_col: str,
    pred_name: str,
    resid_name: str,
) -> tuple[pd.DataFrame, str]:
    spec = spec_by_spec_id("expectation_main")
    eligible = apply_sample_mask(df, spec.sample_key).copy().reset_index(drop=True)
    if y_col not in eligible.columns:
        raise SystemExit(f"missing {y_col} for component residuals")
    cat_cols = [c for c, _ in spec.categorical_cols]
    extra_cols: list[str] = list(spec.extra_controls)
    if spec.use_topic_fe:
        eligible["topic_id"] = eligible["topic_id"].astype("Int64").astype(str)
        cat_cols = [*cat_cols, "topic_id"]
    frame = prepare_model_frame(
        eligible,
        y_col=y_col,
        x_cols=list(spec.x_cols),
        control_cols=spec.control_cols,
        extra_cols=extra_cols or None,
        categorical_cols=cat_cols or None,
    )
    formula = build_formula(spec, y_col=y_col)
    diagnostics = diagnose_model_frame(
        frame,
        formula,
        fitter="ols",
        y_col=y_col,
        x_cols=list(spec.x_cols),
        control_cols=spec.control_cols,
    )
    if diagnostics["zero_variance_cols"]:
        msg = f"Zero-variance in {y_col}: {diagnostics['zero_variance_cols']}. formula={formula!r}"
        raise ZeroVarianceDiagnosticError(msg)
    fit, _n = fit_ols_cluster(frame, formula)
    out = frame.copy()
    out[pred_name] = np.asarray(fit.fittedvalues, dtype=float)
    out[resid_name] = np.asarray(fit.resid, dtype=float)
    if "mid" in eligible.columns:
        out["mid"] = eligible.loc[out.index, "mid"].astype(str).to_numpy()
    summary = format_diagnostic_header(diagnostics, formula=formula) + str(fit.summary())
    return out, summary


def attach_component_residuals(core_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    pieces: dict[str, pd.DataFrame] = {}
    summaries: dict[str, str] = {}
    specs = (
        ("log_engagement", "predicted_log_engagement", "residual_engagement"),
        ("log_likes", "predicted_log_likes", "likes_resid"),
        ("log_comments", "predicted_log_comments", "comments_resid"),
        ("log_reposts", "predicted_log_reposts", "reposts_resid"),
    )
    for y_col, pred, resid in specs:
        scored, summary = fit_y_with_residuals(
            core_df, y_col=y_col, pred_name=pred, resid_name=resid
        )
        pieces[y_col] = scored[["mid", pred, resid]].copy()
        summaries[y_col] = summary
    out = core_df.copy()
    out["mid"] = out["mid"].astype(str)
    for y_col, pred, resid in specs:
        out = out.merge(pieces[y_col], on="mid", how="left")
    out = apply_bc_screens(out)
    return out, summaries


def merge_text(df: pd.DataFrame, text_source: Path | None) -> pd.DataFrame:
    if text_source is None or not text_source.is_file() or df.empty:
        out = df.copy()
        if "analysis_text_final" not in out.columns:
            out["analysis_text_final"] = pd.NA
        return out
    text_df = pd.read_csv(text_source, dtype={"mid": str}, low_memory=False)
    cols = [c for c in ("mid", "analysis_text_final", "text", "text_hash") if c in text_df.columns]
    if "mid" not in cols:
        return df
    slim = text_df[cols].drop_duplicates(subset=["mid"])
    if "analysis_text_final" not in slim.columns and "text" in slim.columns:
        slim = slim.rename(columns={"text": "analysis_text_final"})
    keep = [c for c in ("mid", "analysis_text_final", "text_hash") if c in slim.columns]
    return df.merge(slim[keep], on="mid", how="left", suffixes=("", "_txt"))


def _pool_counts(df: pd.DataFrame, *, indirect: int | None, narratives: tuple[str, ...]) -> dict:
    out: dict[str, int] = {}
    work = df
    if indirect is not None and "indirect_clean" in work.columns:
        work = work.loc[pd.to_numeric(work["indirect_clean"], errors="coerce") == indirect]
    for t2 in (0, 1):
        period = work.loc[pd.to_numeric(work["t2"], errors="coerce") == t2]
        for narr in narratives:
            key = f"t2={t2}|{narr}"
            n = int((period["label_narrative_clean"].astype("string") == narr).sum())
            out[key] = n
    return out


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            export = frame if not frame.empty else pd.DataFrame({"note": ["empty"]})
            export.to_excel(writer, sheet_name=name[:31], index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_MODEL_DATA_FINAL)
    p.add_argument("--text-source", type=Path, default=PATH_ANALYSIS_TOPICS)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--out-dir", type=Path, default=OUT_CASES)
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(
            f"input not found: {inp}\nRun: uv run python scripts/03_prepare_model_data.py"
        )

    df = pd.read_csv(inp, dtype={"mid": str})
    df["mid"] = df["mid"].astype(str)
    text_path = args.text_source.expanduser().resolve() if args.text_source else None
    df = merge_text(df, text_path)

    broader = apply_sample_mask(df, "h1_broader")
    core = apply_sample_mask(df, "h1")

    a1 = draw_a1(broader, seed=args.seed)
    a2_ind = draw_a2_indirect_from_a1(a1, seed=args.seed)
    a2_dir = draw_a2_direct(core, seed=args.seed)

    scored, _summaries = attach_component_residuals(df)
    b = select_b(scored, seed=args.seed)
    c = select_c(scored, seed=args.seed)
    neg = select_contrasts(scored, seed=args.seed)
    unclear = select_unclear(broader, seed=args.seed)

    unique = merge_unique_cases([a1, a2_ind, a2_dir, b, c, neg, unclear])
    unique = enrich_unique_with_scores(unique, scored, b)
    unique_selected = (
        unique.loc[unique["selected"]].copy() if "selected" in unique.columns else unique
    )
    unique_selected = ordered_export(unique_selected)
    log = build_selection_log([a1, a2_ind, a2_dir, b, c, neg, unclear])
    for reason in log["exclusion_reason"].astype(str):
        validate_exclusion_reason(reason)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)

    unique_path = out_dir / "qualitative_matrix.csv"
    log_path = out_dir / "qualitative_selection_log.csv"
    xlsx_path = out_dir / "qualitative_matrix.xlsx"
    unique_selected.to_csv(unique_path, index=False, encoding="utf-8-sig")
    log.to_csv(log_path, index=False, encoding="utf-8-sig")
    write_workbook(
        xlsx_path,
        {
            "unique_cases": unique_selected,
            "a1": ordered_export(a1),
            "a2": ordered_export(pd.concat([a2_ind, a2_dir], ignore_index=True)),
            "b": ordered_export(b),
            "c": ordered_export(c),
            "contrasts": ordered_export(neg),
            "unclear_sensitivity": ordered_export(unclear),
            "selection_log": log,
        },
    )

    a1_sel = a1.loc[a1["selected"]] if not a1.empty else a1
    a2_ind_sel = a2_ind.loc[a2_ind["selected"]] if not a2_ind.empty else a2_ind
    a2_dir_sel = a2_dir.loc[a2_dir["selected"]] if not a2_dir.empty else a2_dir
    a2_ind_mids = set(a2_ind_sel["mid"].astype(str)) if not a2_ind_sel.empty else set()
    a1_core_mids = (
        set(
            a1_sel.loc[
                a1_sel["label_narrative_clean"].astype("string").isin(TRAUMA_NARRATIVES_CLEAN),
                "mid",
            ].astype(str)
        )
        if not a1_sel.empty
        else set()
    )

    n_unique_selected = int(unique_selected.shape[0])
    report: dict[str, Any] = {
        "built_at": utc_now_iso(),
        "semantic_analysis_ids": SEMANTIC_ANALYSIS_IDS,
        "sample_a_note": SAMPLE_A_NOTE,
        "engagement_not_used_for_sample_a": True,
        "allowed_exclusion_reasons": sorted(QUALITATIVE_EXCLUSION_REASONS),
        "a1_quotas": A1_QUOTAS,
        "a2_quotas": A2_QUOTAS,
        "n_unique_rows": int(len(unique)),
        "n_unique_selected": n_unique_selected,
        "n_a1_selected": int(a1_sel.shape[0]),
        "n_a2_indirect_selected": int(a2_ind_sel.shape[0]),
        "n_a2_direct_selected": int(a2_dir_sel.shape[0]),
        "n_b_selected": int(b["selected"].sum()) if not b.empty else 0,
        "n_c_selected": int(c["selected"].sum()) if not c.empty else 0,
        "n_contrast_selected": int(neg["selected"].sum()) if not neg.empty else 0,
        "n_unclear_selected": int(unclear["selected"].sum()) if not unclear.empty else 0,
        "a2_indirect_subset_of_a1": a2_ind_mids <= a1_core_mids,
        "pool_counts_broader_indirect": _pool_counts(
            broader.loc[pd.to_numeric(broader["indirect_clean"], errors="coerce") == 1],
            indirect=None,
            narratives=(NARR_TRAUMA, NARR_MEMORY, NARR_WEATHER, NARR_UNCLEAR),
        ),
        "pool_counts_core_direct": _pool_counts(
            core,
            indirect=0,
            narratives=(NARR_TRAUMA, NARR_MEMORY),
        ),
        "paths": {
            "unique_cases": str(unique_path),
            "selection_log": str(log_path),
            "workbook": str(xlsx_path),
        },
    }
    write_json_report(out_dir / "qualitative_matrix_report.json", report)
    write_json_report(OUT_REPORTS / "qualitative_matrix_report.json", report)
    print(
        f"qualitative matrix -> {out_dir} unique_selected={n_unique_selected} "
        f"A1={report['n_a1_selected']} A2ind={report['n_a2_indirect_selected']} "
        f"A2dir={report['n_a2_direct_selected']}"
    )


if __name__ == "__main__":
    main()
