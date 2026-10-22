"""Sensitivity of the H2 indirect-expression contrast to period-specific misclassification.

The descriptive contrast is the Indirect/Mixed share on the H2 sample
(peripheral × disaster-impact, non-missing expression): about 14.0% in 2021
and 29.4% in 2025. The composition-adjusted logit (H2-M2) has OR ≈ 1.46.

Error rates come from the restricted Core B200 pairs in data/restricted,
restricted to Direct vs Indirect/Mixed and to rows whose LLM label is one of
those two classes (Unclear is missing in the models). Sensitivity and
false-positive rate are allowed to differ by period. Shares are corrected with the Rogan–Gladen
estimator. The H2-M2 odds ratio is re-estimated by drawing a latent label
from the period-specific positive and negative predictive probabilities and
refitting the same logit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PATH_MODEL_DATA_FINAL,
    PROJECT_ROOT,
    apply_sample_mask,
    build_formula,
    prepare_model_frame,
    spec_by_spec_id,
)

VAL_CSV = PROJECT_ROOT / "data" / "restricted" / "expression_validation_pairs.csv"
COUNTS_CSV = PROJECT_ROOT / "data" / "public" / "h2_indirect_counts.csv"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_PUBLIC = PROJECT_ROOT / "outputs" / "public"
OUT_JSON = PROJECT_ROOT / "outputs" / "models" / "misclassification_sensitivity_summary.json"

BINARY = ("Direct", "Indirect-Mixed")
N_IMPUTE = 80
N_BOOT = 200
RNG_SEED = 20261022


def rogan_gladen(p_obs: float, sensitivity: float, false_positive_rate: float) -> float:
    """Invert p = FPR + (Se − FPR) π. Returns NaN when the denominator is not positive."""
    denom = sensitivity - false_positive_rate
    if denom <= 0:
        return float("nan")
    return (p_obs - false_positive_rate) / denom


def crude_or(p_ref: float, p_focal: float) -> float:
    """Odds ratio of two proportions, focal versus reference."""
    if not (0 < p_ref < 1 and 0 < p_focal < 1):
        return float("nan")
    return (p_focal / (1 - p_focal)) / (p_ref / (1 - p_ref))


def nullifying_false_positive_rate(
    p_focal: float, sensitivity_focal: float, pi_ref: float
) -> float:
    """FPR in the focal period that would make its corrected prevalence equal pi_ref."""
    denom = 1.0 - pi_ref
    if denom <= 0:
        return float("nan")
    return (p_focal - pi_ref * sensitivity_focal) / denom


def conditional_probabilities(
    sensitivity: float, false_positive_rate: float, prevalence: float, p_obs: float
) -> tuple[float, float]:
    """Return P(true=1 | labeled 1) and P(true=1 | labeled 0)."""
    if not (0 < p_obs < 1):
        return float("nan"), float("nan")
    ppv = sensitivity * prevalence / p_obs
    p_neg = (1.0 - sensitivity) * prevalence / (1.0 - p_obs)
    return ppv, p_neg


def _binary_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    gold = frame["label_expression"].astype("string").str.strip()
    pred = frame["llm_label_expression"].astype("string").str.strip()
    keep = gold.isin(BINARY) & pred.isin(BINARY)
    out = frame.loc[keep, ["period"]].copy()
    out["gold"] = (gold[keep] == "Indirect-Mixed").astype(int)
    out["pred"] = (pred[keep] == "Indirect-Mixed").astype(int)
    return out


def rates_from_pairs(pairs: pd.DataFrame) -> dict[str, float] | None:
    if pairs.empty:
        return None
    pos = pairs["gold"] == 1
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    sensitivity = float(pairs.loc[pos, "pred"].mean())
    false_positive_rate = float(pairs.loc[neg, "pred"].mean())
    return {
        "n_gold_indirect": n_pos,
        "n_gold_direct": n_neg,
        "sensitivity": sensitivity,
        "false_positive_rate": false_positive_rate,
        "tp": int(pairs.loc[pos, "pred"].sum()),
        "fp": int(pairs.loc[neg, "pred"].sum()),
    }


def period_rates(pairs: pd.DataFrame, period: str) -> dict[str, float]:
    stats = rates_from_pairs(pairs.loc[pairs["period"] == period])
    if stats is None:
        msg = f"no binary validation pairs for {period}"
        raise ValueError(msg)
    return stats


def bootstrap_rates(
    pairs: pd.DataFrame, period: str, rng: np.random.Generator
) -> dict[str, float] | None:
    block = pairs.loc[pairs["period"] == period]
    if len(block) < 2:
        return None
    draw = block.sample(n=len(block), replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
    return rates_from_pairs(draw)


def share_row(
    name: str,
    p1: float,
    p2: float,
    n1: int,
    n2: int,
    se1: float,
    fpr1: float,
    se2: float,
    fpr2: float,
) -> dict[str, Any]:
    pi1 = rogan_gladen(p1, se1, fpr1)
    pi2 = rogan_gladen(p2, se2, fpr2)
    return {
        "scenario": name,
        "se_2021": se1,
        "fpr_2021": fpr1,
        "se_2025": se2,
        "fpr_2025": fpr2,
        "share_2021": pi1,
        "share_2025": pi2,
        "pp_gap": pi2 - pi1,
        "crude_or": crude_or(pi1, pi2),
        "n_2021": n1,
        "n_2025": n2,
        "in_unit_interval": bool(0 < pi1 < 1 and 0 < pi2 < 1),
    }


def _design_matrix(
    df: pd.DataFrame, spec_id: str
) -> tuple[pd.DataFrame, np.ndarray, int, np.ndarray]:
    spec = spec_by_spec_id(spec_id)
    eligible = apply_sample_mask(df, spec.sample_key).copy()
    eligible["topic_id"] = eligible["topic_id"].astype("Int64").astype(str)
    cat_cols = [c for c, _ in spec.categorical_cols]
    if spec.use_topic_fe:
        cat_cols.append("topic_id")
    frame = prepare_model_frame(
        eligible,
        y_col=spec.y_col,
        x_cols=list(spec.x_cols),
        control_cols=spec.control_cols,
        extra_cols=list(spec.extra_controls) or None,
        categorical_cols=cat_cols or None,
    )
    formula = build_formula(spec)
    y_mat, x_mat = patsy.dmatrices(formula, frame, return_type="dataframe")
    groups = frame.loc[x_mat.index, "account_id"].to_numpy()
    t2_col = list(x_mat.columns).index("t2")
    return y_mat, np.asarray(x_mat, dtype=float), t2_col, groups


def fit_t2_or(y_vec: np.ndarray, x_mat: np.ndarray, t2_col: int) -> float:
    model = sm.GLM(y_vec, x_mat, family=Binomial())
    result = model.fit(disp=0, maxiter=100)
    beta = float(np.asarray(result.params)[t2_col])
    return float(np.exp(beta))


def impute_outcome(
    y_obs: np.ndarray,
    t2: np.ndarray,
    rates: dict[int, tuple[float, float]],
    prevalences: dict[int, float],
    observed_p: dict[int, float],
    rng: np.random.Generator,
) -> np.ndarray | None:
    probs = np.empty(len(y_obs), dtype=float)
    for period in (0, 1):
        se, fpr = rates[period]
        pi = prevalences[period]
        ppv, p_false_neg = conditional_probabilities(se, fpr, pi, observed_p[period])
        if not (0 <= ppv <= 1 and 0 <= p_false_neg <= 1):
            return None
        mask = t2 == period
        labeled = y_obs[mask] == 1
        probs[mask] = np.where(labeled, ppv, p_false_neg)
    return rng.random(len(y_obs)) < probs


def published_counts(path: Path = COUNTS_CSV) -> dict[int, tuple[float, int, int]]:
    """Return {t2: (share, n, n_indirect)} from the public H2 count file."""
    frame = pd.read_csv(path)
    out: dict[int, tuple[float, int, int]] = {}
    for period_code, t2 in (("T1", 0), ("T2", 1)):
        row = frame.loc[frame["period"] == period_code].iloc[0]
        n = int(row["n"])
        n_indirect = int(row["n_indirect"])
        out[t2] = (n_indirect / n, n, n_indirect)
    return out


def rates_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Period-specific and pooled binary error rates for the public release."""
    rows = []
    blocks = (
        ("T1", pairs.loc[pairs["period"] == "T1"]),
        ("T2", pairs.loc[pairs["period"] == "T2"]),
    )
    pooled = rates_from_pairs(pairs)
    if pooled is None:
        raise RuntimeError("pooled validation rates are undefined")
    named = {name: rates_from_pairs(block) for name, block in blocks}
    named["pooled"] = pooled
    for name, stats in named.items():
        if stats is None:
            raise RuntimeError(f"validation rates are undefined for {name}")
        rows.append(
            {
                "period": name,
                "n_gold_indirect": stats["n_gold_indirect"],
                "n_gold_direct": stats["n_gold_direct"],
                "true_positive": stats["tp"],
                "false_positive": stats["fp"],
                "sensitivity": stats["sensitivity"],
                "false_positive_rate": stats["false_positive_rate"],
            }
        )
    return pd.DataFrame(rows)


def observed_period_shares(y_obs: np.ndarray, t2: np.ndarray) -> dict[int, tuple[float, int]]:
    out: dict[int, tuple[float, int]] = {}
    for period in (0, 1):
        mask = t2 == period
        n = int(mask.sum())
        out[period] = (float(y_obs[mask].mean()), n)
    return out


def summarize_ors(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n_reps": 0, "median_or": float("nan"), "p2_5": float("nan"), "p97_5": float("nan")}
    return {
        "n_reps": int(len(arr)),
        "median_or": float(np.median(arr)),
        "p2_5": float(np.quantile(arr, 0.025)),
        "p97_5": float(np.quantile(arr, 0.975)),
        "share_or_above_1": float(np.mean(arr > 1)),
    }


def _refit_h2_odds_ratios(
    model_df: pd.DataFrame,
    pairs: pd.DataFrame,
    se1: float,
    fpr1: float,
    se2: float,
    fpr2: float,
    se_pool: float,
    fpr_pool: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Impute latent H2-M2 labels and refit. Requires the restricted model table."""
    y_m2, x_m2, t2_col_m2, groups_m2 = _design_matrix(model_df, "h2_m2_composition")
    y_m1, x_m1, t2_col_m1, _groups_m1 = _design_matrix(model_df, "h2_m1_period")
    y_obs_m2 = np.asarray(y_m2).ravel()
    frame_index = y_m2.index
    t2_m2 = np.rint(x_m2[:, t2_col_m2]).astype(int)
    est_shares = observed_period_shares(y_obs_m2, t2_m2)

    observed_or_m2 = fit_t2_or(y_obs_m2, x_m2, t2_col_m2)
    observed_or_m1 = fit_t2_or(np.asarray(y_m1).ravel(), x_m1, t2_col_m1)
    cluster_fit = sm.GLM(y_obs_m2, x_m2, family=Binomial()).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": groups_m2}
    )
    beta = float(np.asarray(cluster_fit.params)[t2_col_m2])
    se_beta = float(np.asarray(cluster_fit.bse)[t2_col_m2])
    cluster_or_ci = (float(np.exp(beta - 1.96 * se_beta)), float(np.exp(beta + 1.96 * se_beta)))

    rng = np.random.default_rng(RNG_SEED)

    def impute_and_fit(
        se_a: float,
        fpr_a: float,
        se_b: float,
        fpr_b: float,
        local_rng: np.random.Generator,
    ) -> float | None:
        p_est = {0: est_shares[0][0], 1: est_shares[1][0]}
        pi = {
            0: rogan_gladen(p_est[0], se_a, fpr_a),
            1: rogan_gladen(p_est[1], se_b, fpr_b),
        }
        if not (0 < pi[0] < 1 and 0 < pi[1] < 1):
            return None
        y_draw = impute_outcome(
            y_obs_m2,
            t2_m2,
            {0: (se_a, fpr_a), 1: (se_b, fpr_b)},
            pi,
            p_est,
            local_rng,
        )
        if y_draw is None or y_draw.sum() == 0 or y_draw.sum() == len(y_draw):
            return None
        try:
            return fit_t2_or(y_draw.astype(float), x_m2, t2_col_m2)
        except (np.linalg.LinAlgError, ValueError):
            return None

    def collect_fixed(
        se_a: float, fpr_a: float, se_b: float, fpr_b: float, n_rep: int
    ) -> list[float]:
        out: list[float] = []
        for _ in range(n_rep):
            value = impute_and_fit(se_a, fpr_a, se_b, fpr_b, rng)
            if value is not None and np.isfinite(value):
                out.append(value)
        return out

    fixed_draws = {
        "validation_differential": collect_fixed(se1, fpr1, se2, fpr2, N_IMPUTE),
        "pooled_nondifferential": collect_fixed(se_pool, fpr_pool, se_pool, fpr_pool, N_IMPUTE),
        "sensitivity_0.80_validation_fpr": collect_fixed(0.80, fpr1, 0.80, fpr2, N_IMPUTE),
    }
    boot_draws: list[float] = []
    for _ in range(N_BOOT):
        b1 = bootstrap_rates(pairs, "T1", rng)
        b2 = bootstrap_rates(pairs, "T2", rng)
        if b1 is None or b2 is None:
            continue
        value = impute_and_fit(
            b1["sensitivity"],
            b1["false_positive_rate"],
            b2["sensitivity"],
            b2["false_positive_rate"],
            rng,
        )
        if value is not None and np.isfinite(value):
            boot_draws.append(value)

    or_rows: list[dict[str, Any]] = []
    for name, draws in fixed_draws.items():
        row = {"scenario": name, "uncertainty": "imputation_at_fixed_rates"}
        row.update(summarize_ors(draws))
        or_rows.append(row)
    boot_row = {
        "scenario": "validation_differential",
        "uncertainty": "bootstrap_rates_and_imputation",
    }
    boot_row.update(summarize_ors(boot_draws))
    or_rows.append(boot_row)
    estimation = {
        "or_refit": "h2_m2_imputation",
        "estimation_n_h2_m2": int(len(y_obs_m2)),
        "estimation_shares": {
            "2021": est_shares[0][0],
            "2025": est_shares[1][0],
            "n_2021": est_shares[0][1],
            "n_2025": est_shares[1][1],
        },
        "observed_or_h2_m1": observed_or_m1,
        "observed_or_h2_m2": observed_or_m2,
        "observed_or_h2_m2_cluster_ci95": list(cluster_or_ci),
        "design_index_matches_frame": bool(frame_index.equals(y_m2.index)),
    }
    return or_rows, estimation


def _descriptive_shares_from_model(model_df: pd.DataFrame) -> dict[int, tuple[float, int, int]]:
    descriptive = apply_sample_mask(model_df, "h2").dropna(subset=["indirect_clean", "t2"])
    descriptive["t2"] = pd.to_numeric(descriptive["t2"], errors="coerce")
    y_desc = pd.to_numeric(descriptive["indirect_clean"], errors="coerce").to_numpy()
    t2_desc = descriptive["t2"].to_numpy(dtype=int)
    shares = observed_period_shares(y_desc, t2_desc)
    out: dict[int, tuple[float, int, int]] = {}
    for period in (0, 1):
        share, n = shares[period]
        out[period] = (share, n, int(y_desc[t2_desc == period].sum()))
    return out


def run() -> dict[str, Any]:
    if not VAL_CSV.is_file():
        msg = (
            "Restricted validation pairs are not published. "
            "Place them at data/restricted/expression_validation_pairs.csv to recompute. "
            "Public aggregates remain in outputs/public/."
        )
        raise FileNotFoundError(msg)
    validation = pd.read_csv(VAL_CSV, encoding="utf-8-sig")
    pairs = _binary_pairs(validation)
    rates_2021 = period_rates(pairs, "T1")
    rates_2025 = period_rates(pairs, "T2")
    pooled = rates_from_pairs(pairs)
    if pooled is None:
        raise RuntimeError("pooled validation rates are undefined")

    frozen = published_counts()
    model_available = PATH_MODEL_DATA_FINAL.is_file()
    model_df = pd.read_csv(PATH_MODEL_DATA_FINAL, low_memory=False) if model_available else None
    if model_df is not None:
        checked = _descriptive_shares_from_model(model_df)
        for period in (0, 1):
            if checked[period][1:] != frozen[period][1:]:
                msg = f"H2 counts for t2={period} differ from {COUNTS_CSV.name}"
                raise RuntimeError(msg)
        desc_shares = checked
    else:
        desc_shares = frozen
    p1, n1, _k1 = desc_shares[0]
    p2, n2, _k2 = desc_shares[1]

    se1 = rates_2021["sensitivity"]
    fpr1 = rates_2021["false_positive_rate"]
    se2 = rates_2025["sensitivity"]
    fpr2 = rates_2025["false_positive_rate"]
    se_pool = pooled["sensitivity"]
    fpr_pool = pooled["false_positive_rate"]

    share_rows = [
        share_row("observed", p1, p2, n1, n2, 1.0, 0.0, 1.0, 0.0),
        share_row("validation_differential", p1, p2, n1, n2, se1, fpr1, se2, fpr2),
        share_row("pooled_nondifferential", p1, p2, n1, n2, se_pool, fpr_pool, se_pool, fpr_pool),
        share_row("sensitivity_0.80_validation_fpr", p1, p2, n1, n2, 0.80, fpr1, 0.80, fpr2),
        share_row("sensitivity_0.95_validation_fpr", p1, p2, n1, n2, 0.95, fpr1, 0.95, fpr2),
    ]
    pi1_primary = share_rows[1]["share_2021"]
    fpr2_null = nullifying_false_positive_rate(p2, se2, pi1_primary)

    if model_df is None:
        or_rows = []
        estimation: dict[str, Any] = {"or_refit": "skipped_restricted_model_data_absent"}
    else:
        or_rows, estimation = _refit_h2_odds_ratios(
            model_df,
            pairs,
            se1,
            fpr1,
            se2,
            fpr2,
            se_pool,
            fpr_pool,
        )

    summary: dict[str, Any] = {
        "validation_csv": "data/restricted/expression_validation_pairs.csv",
        "counts_csv": "data/public/h2_indirect_counts.csv",
        "binary_definition": "Direct vs Indirect-Mixed, both gold and LLM; Unclear excluded",
        "rates_2021": rates_2021,
        "rates_2025": rates_2025,
        "rates_pooled": pooled,
        "descriptive_n": {"2021": n1, "2025": n2, "total": n1 + n2},
        "observed_shares": {"2021": p1, "2025": p2, "crude_or": crude_or(p1, p2)},
        "nullifying_fpr_2025": fpr2_null,
        "share_scenarios": share_rows,
        "or_scenarios": or_rows,
        "n_impute_fixed": N_IMPUTE,
        "n_boot": N_BOOT,
        "seed": RNG_SEED,
    }
    summary.update(estimation)

    rate_frame = rates_table(pairs)
    share_frame = pd.DataFrame(share_rows)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_PUBLIC.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    for folder in (OUT_TABLES, OUT_PUBLIC):
        share_frame.to_csv(folder / "table_misclassification_sensitivity_shares.csv", index=False)
        rate_frame.to_csv(folder / "table_expression_validation_period_rates.csv", index=False)
        if or_rows:
            pd.DataFrame(or_rows).to_csv(
                folder / "table_misclassification_sensitivity_or.csv", index=False
            )
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    try:
        summary = run()
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(2) from exc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
