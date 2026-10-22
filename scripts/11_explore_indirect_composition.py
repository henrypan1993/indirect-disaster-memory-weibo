"""Exploratory composition of peripheral Indirect/Mixed posts (2021 vs 2025).

RQ1: narrative / emotional orientation shifts (incl. Memory-Reactivation share).
RQ2: composition-based correlational explanation of engagement differences.

Unlike H1/H2 (trauma-related core), this sample is Peripheral × Indirect/Mixed
without requiring disaster_impact_related, to retain heterogeneous narratives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    EMOTION_POLARITY_MAP,
    PATH_MODEL_DATA_FINAL,
    PERIPHERAL_SAMPLE_CONTROLS,
    PROJECT_ROOT,
    REF_EMOTION,
    REF_NARRATIVE,
    REF_TOPIC,
    ModelSpec,
    ZeroVarianceDiagnosticError,
    build_formula,
    coef_row,
    diagnose_model_frame,
    emotion_polarity_series,
    fit_ols_cluster,
    format_diagnostic_header,
    prepare_model_frame,
    utc_now_iso,
    write_json_report,
)

OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"
OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"

COLOR_T1 = "#6b9ac4"
COLOR_T2 = "#d98a5f"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

SAMPLE_NOTE = (
    "Unlike the hypothesis tests focusing on trauma-related posts, this "
    "exploratory analysis examines the broader composition of peripheral "
    "indirect expressions to characterize heterogeneous narrative orientations."
)


def select_peripheral_indirect(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[
        (df["include_main"] == 1)
        & (df["peripheral"] == 1)
        & (df["indirect_clean"] == 1)
    ].copy()
    is_mem = out["label_narrative_clean"].astype("string") == "Memory-Reactivation"
    out["memory_reactivation"] = is_mem.fillna(False).astype(int)
    out["emotion_polarity"] = emotion_polarity_series(out["label_emotion_clean"])
    # Keep unmapped for composition tables
    emo_raw = out["label_emotion_clean"].astype("string")
    out["emotion_polarity_table"] = emo_raw.map(EMOTION_POLARITY_MAP)
    unmapped = out["emotion_polarity_table"].isna() & emo_raw.notna()
    out.loc[unmapped, "emotion_polarity_table"] = "other"
    out.loc[emo_raw.isna(), "emotion_polarity_table"] = "missing"
    return out


def composition_long_table(
    df: pd.DataFrame,
    row_col: str,
    *,
    period_col: str = "t2",
) -> pd.DataFrame:
    """Counts, row %, and column % for row_col × period."""
    ct = pd.crosstab(df[row_col], df[period_col], dropna=False)
    # Ensure both periods present as columns
    for p in (0, 1):
        if p not in ct.columns:
            ct[p] = 0
    ct = ct[[0, 1]].sort_index()
    row_tot = ct.sum(axis=1).replace(0, np.nan)
    col_tot = ct.sum(axis=0).replace(0, np.nan)
    rows: list[dict[str, Any]] = []
    for cat in ct.index:
        for period, year_label in ((0, "2021_T1"), (1, "2025_T2")):
            n = int(ct.loc[cat, period])
            rows.append(
                {
                    "category": str(cat),
                    "t2": period,
                    "year_label": year_label,
                    "n": n,
                    "row_pct": (
                        float(n / row_tot.loc[cat])
                        if pd.notna(row_tot.loc[cat])
                        else np.nan
                    ),
                    "col_pct": (
                        float(n / col_tot.loc[period])
                        if pd.notna(col_tot.loc[period])
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def cramers_v(chi2: float, n: int, n_rows: int, n_cols: int) -> float:
    if n <= 0 or min(n_rows, n_cols) <= 1:
        return float("nan")
    return float(np.sqrt(chi2 / (n * (min(n_rows, n_cols) - 1))))


def _pivot_col_pct(tbl: pd.DataFrame, categories: list[str] | None = None) -> pd.DataFrame:
    """Wide matrix: index=category, columns=year_label, values=col_pct."""
    sub = tbl.copy()
    sub["category"] = sub["category"].astype(str)
    # Drop empty / nan category rows for plotting
    sub = sub.loc[~sub["category"].isin(["nan", "None", ""])]
    if categories is not None:
        sub = sub.loc[sub["category"].isin(categories)]
    wide = sub.pivot_table(
        index="category",
        columns="year_label",
        values="col_pct",
        aggfunc="first",
    )
    for col in ("2021_T1", "2025_T2"):
        if col not in wide.columns:
            wide[col] = 0.0
    wide = wide[["2021_T1", "2025_T2"]].fillna(0.0)
    if categories is not None:
        wide = wide.reindex([c for c in categories if c in wide.index])
    else:
        # Sort by mean share descending
        wide = wide.loc[wide.mean(axis=1).sort_values(ascending=False).index]
    return wide


def _grouped_bars(
    ax: plt.Axes,
    wide: pd.DataFrame,
    *,
    title: str,
    ylabel: str = "Share within year",
    annotate: bool = True,
) -> None:
    cats = list(wide.index.astype(str))
    x = np.arange(len(cats))
    width = 0.36
    t1 = wide["2021_T1"].to_numpy(dtype=float)
    t2 = wide["2025_T2"].to_numpy(dtype=float)
    bars1 = ax.bar(x - width / 2, t1, width, label="2021 (T1)", color=COLOR_T1)
    bars2 = ax.bar(x + width / 2, t2, width, label="2025 (T2)", color=COLOR_T2)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, min(1.05, max(t1.max(), t2.max(), 0.05) * 1.25))
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if annotate:
        for bars in (bars1, bars2):
            for bar in bars:
                h = bar.get_height()
                if h <= 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )


def _chi2_subtitle(chi2_path: Path, variable: str) -> str:
    if not chi2_path.is_file():
        return ""
    payload = json.loads(chi2_path.read_text(encoding="utf-8"))
    for test in payload.get("tests", []):
        if test.get("variable") == variable:
            p = test.get("pvalue")
            v = test.get("cramers_v")
            if p is None or v is None or (isinstance(p, float) and np.isnan(p)):
                return ""
            p_txt = "<.001" if p < 0.001 else f"={p:.3f}"
            return f"χ² p{p_txt}, Cramér's V={v:.2f}"
    return ""


def write_composition_figures(
    mem_tbl: pd.DataFrame,
    narr_tbl: pd.DataFrame,
    emo_tbl: pd.DataFrame,
    *,
    chi2_path: Path,
    out_dir: Path,
    n_t1: int,
    n_t2: int,
) -> list[Path]:
    """Save grouped-bar composition figures (2021 vs 2025)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Memory-Reactivation share
    mem = mem_tbl.copy()
    mem["category"] = mem["category"].astype(str).map(
        {"0": "Other narrative", "1": "Memory-Reactivation"}
    )
    mem_wide = _pivot_col_pct(mem, ["Memory-Reactivation", "Other narrative"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    sub = _chi2_subtitle(chi2_path, "memory_reactivation")
    title = "Memory-Reactivation narrative category\nPeripheral × Indirect/Mixed"
    if sub:
        title = f"{title}\n{sub}"
    _grouped_bars(ax, mem_wide, title=title)
    ax.set_xlabel("")
    path = out_dir / "indirect_composition_memory_by_year.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    # Narrative orientation
    narr_order = [
        "Trauma-Help-Loss",
        "Memory-Reactivation",
        "Restoration",
        "Ordinary-Weather-Info",
        "Other-Unclear",
    ]
    narr_wide = _pivot_col_pct(narr_tbl, narr_order)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    sub = _chi2_subtitle(chi2_path, "label_narrative_clean")
    title = "Narrative orientation by year\nPeripheral × Indirect/Mixed"
    if sub:
        title = f"{title}\n{sub}"
    _grouped_bars(ax, narr_wide, title=title)
    path = out_dir / "indirect_composition_narrative_by_year.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    # Emotion polarity (drop missing if tiny)
    emo_order = ["negative", "neutral", "mixed", "positive", "other"]
    emo_wide = _pivot_col_pct(emo_tbl, emo_order)
    emo_wide = emo_wide.loc[emo_wide.sum(axis=1) > 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sub = _chi2_subtitle(chi2_path, "emotion_polarity_table")
    title = "Emotional orientation by year\nPeripheral × Indirect/Mixed"
    if sub:
        title = f"{title}\n{sub}"
    _grouped_bars(ax, emo_wide, title=title)
    path = out_dir / "indirect_composition_emotion_by_year.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    # Combined three-panel overview
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    _grouped_bars(
        axes[0],
        mem_wide,
        title="(A) Memory-Reactivation",
        annotate=True,
    )
    _grouped_bars(
        axes[1],
        narr_wide,
        title="(B) Narrative orientation",
        annotate=False,
    )
    _grouped_bars(
        axes[2],
        emo_wide,
        title="(C) Emotion polarity",
        annotate=False,
    )
    fig.suptitle(
        f"Composition of peripheral Indirect/Mixed posts "
        f"(N_T1={n_t1:,}; N_T2={n_t2:,})",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    path = out_dir / "indirect_composition_overview.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)
    return written


def chi2_report(df: pd.DataFrame, row_col: str, period_col: str = "t2") -> dict[str, Any]:
    ct = pd.crosstab(df[row_col], df[period_col], dropna=False)
    # Drop all-zero rows/cols for chi2
    ct = ct.loc[(ct.sum(axis=1) > 0), :]
    ct = ct.loc[:, (ct.sum(axis=0) > 0)]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {
            "variable": row_col,
            "chi2": np.nan,
            "df": np.nan,
            "pvalue": np.nan,
            "cramers_v": np.nan,
            "n": int(ct.to_numpy().sum()) if ct.size else 0,
            "n_rows": int(ct.shape[0]),
            "n_cols": int(ct.shape[1]),
            "note": "insufficient table dimensions for chi-square",
        }
    chi2, p, dof, _ = chi2_contingency(ct.to_numpy())
    n = int(ct.to_numpy().sum())
    return {
        "variable": row_col,
        "chi2": float(chi2),
        "df": int(dof),
        "pvalue": float(p),
        "cramers_v": cramers_v(float(chi2), n, ct.shape[0], ct.shape[1]),
        "n": n,
        "n_rows": int(ct.shape[0]),
        "n_cols": int(ct.shape[1]),
    }


def _m2a_spec() -> ModelSpec:
    return ModelSpec(
        model_id="indirect_composition_engagement",
        module="Explore",
        y_col="log_engagement",
        x_cols=("memory_reactivation", "t2"),
        sample_key="h1",  # unused; we pass a pre-filtered frame
        fitter="ols",
        focal_terms=("memory_reactivation",),
        control_cols=PERIPHERAL_SAMPLE_CONTROLS,
        use_topic_fe=True,
        categorical_cols=(("emotion_polarity", "neutral"),),
        extra_report_terms=("t2",),
        formal_label="Explore",
        default_spec_id="m2a_reactivation",
    )


def _m2b_spec() -> ModelSpec:
    return ModelSpec(
        model_id="indirect_composition_engagement",
        module="Explore",
        y_col="log_engagement",
        x_cols=("t2",),
        sample_key="h1",
        fitter="ols",
        focal_terms=("t2",),
        control_cols=PERIPHERAL_SAMPLE_CONTROLS,
        use_topic_fe=True,
        categorical_cols=(
            ("label_narrative_clean", REF_NARRATIVE),
            ("emotion_polarity", "neutral"),
        ),
        extra_report_terms=("t2",),
        formal_label="Explore",
        default_spec_id="m2b_narrative",
    )


def _m2c_spec() -> ModelSpec:
    """Interaction model without emotion (plan default)."""
    return ModelSpec(
        model_id="indirect_composition_engagement",
        module="Explore",
        y_col="log_engagement",
        x_cols=("memory_reactivation", "t2"),
        sample_key="h1",
        fitter="ols",
        focal_terms=("memory_reactivation",),
        control_cols=PERIPHERAL_SAMPLE_CONTROLS,
        use_topic_fe=True,
        extra_report_terms=("t2",),
        formal_label="Explore",
        default_spec_id="m2c_reactivation_x_t2",
    )


def fit_explore_ols(
    df: pd.DataFrame,
    spec: ModelSpec,
    *,
    formula_override: str | None = None,
    require_emotion: bool = True,
) -> tuple[list[dict[str, Any]], str, int, str]:
    """Fit OLS on a pre-filtered Peripheral×Indirect frame."""
    sub = df.copy()
    if require_emotion:
        sub = sub.loc[sub["emotion_polarity"].notna()].copy()
    cat_cols = [c for c, _ in spec.categorical_cols]
    if spec.use_topic_fe:
        sub["topic_id"] = sub["topic_id"].astype("Int64").astype(str)
        cat_cols = [*cat_cols, "topic_id"]
    y = spec.y_col
    x_cols = list(spec.x_cols)
    extra = list(spec.extra_controls)
    frame = prepare_model_frame(
        sub,
        y_col=y,
        x_cols=x_cols,
        control_cols=spec.control_cols,
        extra_cols=extra or None,
        categorical_cols=cat_cols or None,
    )
    # Interaction needs both columns present
    if formula_override and "memory_reactivation:t2" in formula_override:
        for c in ("memory_reactivation", "t2"):
            if c not in frame.columns:
                raise ValueError(f"missing {c} for interaction model")
    formula = formula_override or build_formula(spec, y_col=y)
    diagnostics = diagnose_model_frame(
        frame,
        formula,
        fitter="ols",
        y_col=y,
        x_cols=x_cols,
        control_cols=spec.control_cols,
    )
    if diagnostics["zero_variance_cols"]:
        raise ZeroVarianceDiagnosticError(
            f"Zero-variance in {spec.default_spec_id}: {diagnostics['zero_variance_cols']}"
        )
    fit, n = fit_ols_cluster(frame, formula)
    summary = format_diagnostic_header(diagnostics, formula=formula) + str(fit.summary())

    rows: list[dict[str, Any]] = []
    # Report focal + extra + interaction if present
    report_terms = list(spec.focal_terms) + list(spec.extra_report_terms)
    if formula_override and "memory_reactivation:t2" in formula_override:
        report_terms.append("memory_reactivation:t2")
    # Also report emotion / narrative FE coefficients that appear in params
    for term in fit.params.index:
        if term == "Intercept":
            continue
        if term in report_terms or term.startswith("C(") or ":" in term:
            if term not in report_terms:
                report_terms.append(term)
    seen: set[str] = set()
    for term in report_terms:
        if term in seen or term not in fit.params.index:
            continue
        seen.add(term)
        row = coef_row(
            spec.model_id,
            fit,
            term,
            n,
            spec.default_spec_id,
            module=spec.module,
        )
        row["formal_label"] = spec.formal_label
        rows.append(row)
    return rows, summary, n, formula


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_MODEL_DATA_FINAL)
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(
            f"input not found: {inp}\nRun: uv run python scripts/03_prepare_model_data.py"
        )

    df = pd.read_csv(inp, dtype={"mid": str})
    sample = select_peripheral_indirect(df)
    n_total = len(sample)
    n_t1 = int((sample["t2"] == 0).sum())
    n_t2 = int((sample["t2"] == 1).sum())

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)

    # --- Analysis 1 ---
    mem_tbl = composition_long_table(sample, "memory_reactivation")
    narr_tbl = composition_long_table(sample, "label_narrative_clean")
    emo_tbl = composition_long_table(sample, "emotion_polarity_table")

    mem_tbl.to_csv(
        OUT_TABLES / "indirect_composition_memory_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    narr_tbl.to_csv(
        OUT_TABLES / "indirect_composition_narrative_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    emo_tbl.to_csv(
        OUT_TABLES / "indirect_composition_emotion_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    chi2_payload = {
        "built_at": utc_now_iso(),
        "sample_n": n_total,
        "n_t1": n_t1,
        "n_t2": n_t2,
        "tests": [
            chi2_report(sample, "memory_reactivation"),
            chi2_report(sample, "label_narrative_clean"),
            chi2_report(sample, "emotion_polarity_table"),
        ],
        "sample_note": SAMPLE_NOTE,
    }
    chi2_path = OUT_TABLES / "indirect_composition_chi2.json"
    write_json_report(chi2_path, chi2_payload)

    fig_paths = write_composition_figures(
        mem_tbl,
        narr_tbl,
        emo_tbl,
        chi2_path=chi2_path,
        out_dir=OUT_FIGURES,
        n_t1=n_t1,
        n_t2=n_t2,
    )

    # --- Analysis 2 ---
    coef_rows: list[dict[str, Any]] = []
    model_ns: dict[str, int] = {}
    formulas: dict[str, str] = {}

    m2a = _m2a_spec()
    rows_a, sum_a, n_a, form_a = fit_explore_ols(sample, m2a, require_emotion=True)
    coef_rows.extend(rows_a)
    model_ns["m2a"] = n_a
    formulas["m2a"] = form_a
    (OUT_MODELS / "indirect_composition_engagement_m2a.txt").write_text(sum_a, encoding="utf-8")

    m2b = _m2b_spec()
    rows_b, sum_b, n_b, form_b = fit_explore_ols(sample, m2b, require_emotion=True)
    coef_rows.extend(rows_b)
    model_ns["m2b"] = n_b
    formulas["m2b"] = form_b
    (OUT_MODELS / "indirect_composition_engagement_m2b.txt").write_text(sum_b, encoding="utf-8")

    m2c = _m2c_spec()
    # Explicit interaction formula (no emotion, per plan)
    form_c = (
        "log_engagement ~ memory_reactivation + t2 + memory_reactivation:t2 + "
        + " + ".join(PERIPHERAL_SAMPLE_CONTROLS)
        + f" + C(topic_id, Treatment(reference='{REF_TOPIC}'))"
    )
    rows_c, sum_c, n_c, form_c_used = fit_explore_ols(
        sample,
        m2c,
        formula_override=form_c,
        require_emotion=False,
    )
    coef_rows.extend(rows_c)
    model_ns["m2c"] = n_c
    formulas["m2c"] = form_c_used
    (OUT_MODELS / "indirect_composition_engagement_m2c.txt").write_text(sum_c, encoding="utf-8")

    coef_df = pd.DataFrame(coef_rows)
    coef_df["built_at"] = utc_now_iso()
    coef_path = OUT_MODELS / "indirect_composition_engagement_summary.csv"
    coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")

    report = {
        "built_at": utc_now_iso(),
        "role": "exploratory_correlational_composition_analysis",
        "sample_definition": (
            "include_main & peripheral & indirect_clean (no disaster_impact filter)"
        ),
        "sample_note": SAMPLE_NOTE,
        "n_total": n_total,
        "n_t1_2021": n_t1,
        "n_t2_2025": n_t2,
        "emotion_polarity_map": EMOTION_POLARITY_MAP,
        "ref_narrative": REF_NARRATIVE,
        "ref_emotion_polarity": "neutral",
        "ref_emotion_label_unused": REF_EMOTION,
        "formulas": formulas,
        "model_n": model_ns,
        "chi2_path": str(chi2_path),
        "coef_summary_path": str(coef_path),
        "figure_paths": [str(p) for p in fig_paths],
        "wording": (
            "composition-based explanation of engagement differences; "
            "not causal mechanism identification"
        ),
    }
    write_json_report(OUT_REPORTS / "indirect_composition_explore_report.json", report)

    print(
        f"indirect composition explore -> n={n_total} (T1={n_t1}, T2={n_t2}); "
        f"M2a/b/c N={model_ns['m2a']}/{model_ns['m2b']}/{model_ns['m2c']}; "
        f"figures={len(fig_paths)}"
    )


if __name__ == "__main__":
    main()
