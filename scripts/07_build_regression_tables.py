"""Build publication-style regression tables from all_models_summary.csv.

Formal models only: H1, H2 (M1/M2), D1 (M1/M2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, utc_now_iso

IN_ALL = PROJECT_ROOT / "outputs" / "models" / "all_models_summary.csv"
IN_MAIN = PROJECT_ROOT / "outputs" / "models" / "main_models_summary.csv"
IN_APPENDIX = PROJECT_ROOT / "outputs" / "models" / "appendix_without_t2.csv"
IN_ROBUST = PROJECT_ROOT / "outputs" / "models" / "robustness_models_summary.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"

FOCAL_BY_MODEL: dict[str, tuple[str, ...]] = {
    "h1_engagement_indirect": ("indirect_clean",),
    "h2_indirect_period": ("t2",),
    "d1_indirect_peripheral": ("peripheral",),
}

LOGIT_MODELS = frozenset({"h2_indirect_period", "d1_indirect_peripheral"})

MODEL_META: dict[str, dict[str, str]] = {
    "h1_engagement_indirect": {
        "module": "H1",
        "dv_en": "log(Engagement)",
        "dv_zh": "对数互动",
        "sample_en": "Peripheral subsample",
        "sample_zh": "边缘子样本",
        "estimator": "OLS, cluster SE by account",
        "formula_zh": "log_engagement ~ indirect + T2 + controls + topic FE",
    },
    "h2_indirect_period": {
        "module": "H2",
        "dv_en": "Indirect (binary)",
        "dv_zh": "间接/混合表达",
        "sample_en": "Disaster-impact ∩ Peripheral",
        "sample_zh": "灾害影响∩边缘",
        "estimator": "Logit, cluster SE by account",
        "formula_zh": "indirect ~ T2 + controls + topic FE",
    },
    "d1_indirect_peripheral": {
        "module": "D1",
        "dv_en": "Indirect (binary)",
        "dv_zh": "间接/混合表达",
        "sample_en": "Full Core (valid indirect)",
        "sample_zh": "全样本 Core",
        "estimator": "Logit, cluster SE by account",
        "formula_zh": "indirect ~ peripheral + T2 + controls + topic FE",
    },
}

FORMULA_WITHOUT_T2: dict[str, str] = {
    "h1_engagement_indirect": "log_engagement ~ indirect + controls + topic FE",
    "d1_indirect_peripheral": "indirect ~ peripheral + controls + topic FE",
}

TERM_LABEL: dict[str, dict[str, str]] = {
    "indirect_clean": {"en": "Indirect", "zh": "间接表达 (Indirect)"},
    "t2": {"en": "T2 (reactivation)", "zh": "再激活期 (T2)"},
    "peripheral": {"en": "Peripheral", "zh": "边缘位置 (Peripheral)"},
}

SPEC_LABEL_ZH: dict[str, str] = {
    "main": "主结果",
    "h2_m1_period": "H2-M1 基线",
    "h2_m2_composition": "H2-M2 组合调整",
    "d1_m1_total_association": "D1-M1 总体关联",
    "d1_m2_conditional_threshold": "D1-M2 条件阈值",
    "without_t2_control": "敏感性：未控制 T2",
    "engagement_likes": "稳健性：因变量 log(点赞)",
    "engagement_comments": "稳健性：因变量 log(评论)",
    "engagement_reposts": "稳健性：因变量 log(转发)",
    "peripheral_p80": "稳健性：peripheral 阈值 p80",
    "peripheral_p90": "稳健性：peripheral 阈值 p90",
    "peripheral_p95": "稳健性：peripheral 阈值 p95",
}

MODULE_ORDER = [
    "h1_engagement_indirect",
    "h2_indirect_period",
    "d1_indirect_peripheral",
]


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_coef_cell(coef: float, se: float, p: float, decimals: int = 3) -> str:
    return f"{coef:.{decimals}f}{stars(p)} ({se:.{decimals}f})"


def formula_for_spec(model_id: str, spec_id: str) -> str:
    if spec_id == "without_t2_control" or str(spec_id).endswith("_without_t2"):
        return FORMULA_WITHOUT_T2.get(model_id, MODEL_META.get(model_id, {}).get("formula_zh", ""))
    if spec_id == "h2_m2_composition":
        return "indirect ~ T2 + narrative + emotion + relative_window_day + controls + topic FE"
    if spec_id == "d1_m2_conditional_threshold":
        return "indirect ~ peripheral + T2 + verified + log_followers + controls + topic FE"
    return MODEL_META.get(model_id, {}).get("formula_zh", "")


def as_int64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def fmt_n_display(s: pd.Series) -> pd.Series:
    def one(v: object) -> str:
        if pd.isna(v):
            return ""
        return f"{int(v):,}"

    return s.map(one)


def load_summary(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    parts: list[pd.DataFrame] = []
    for p in (IN_MAIN, IN_APPENDIX, IN_ROBUST):
        if p.is_file():
            parts.append(pd.read_csv(p))
    if not parts:
        raise SystemExit("no model summary CSV found; run 04 and 05 with --merge first")
    return pd.concat(parts, ignore_index=True)


def enrich_row(row: pd.Series) -> dict:
    mid = row["model_id"]
    meta = MODEL_META.get(mid, {})
    term = row["term"]
    tlab = TERM_LABEL.get(term, {"en": term, "zh": term})
    coef, se, p = float(row["coef"]), float(row["se"]), float(row["pvalue"])
    out = {
        "model_id": mid,
        "module": row.get("module", meta.get("module", "")),
        "spec_id": row["spec_id"],
        "spec_label_zh": SPEC_LABEL_ZH.get(row["spec_id"], row["spec_id"]),
        "term": term,
        "predictor_en": tlab["en"],
        "predictor_zh": tlab["zh"],
        "dv_en": meta.get("dv_en", ""),
        "dv_zh": meta.get("dv_zh", ""),
        "sample_en": meta.get("sample_en", ""),
        "sample_zh": meta.get("sample_zh", ""),
        "estimator": meta.get("estimator", ""),
        "formula_zh": formula_for_spec(mid, row["spec_id"]),
        "n": int(row["n"]),
        "coef": coef,
        "se": se,
        "pvalue": p,
        "p_fmt": fmt_p(p),
        "stars": stars(p),
        "coef_display": fmt_coef_cell(coef, se, p),
        "notes": row.get("notes", "") if pd.notna(row.get("notes", "")) else "",
        "is_focal": term in FOCAL_BY_MODEL.get(mid, ()),
        "is_logit": mid in LOGIT_MODELS,
    }
    if out["is_logit"]:
        out["odds_ratio"] = float(np.exp(coef))
        out["or_display"] = f"{out['odds_ratio']:.2f}{stars(p)}"
    else:
        out["odds_ratio"] = np.nan
        out["or_display"] = ""
        if mid == "h1_engagement_indirect" and term == "indirect_clean":
            pct = (np.exp(coef) - 1) * 100
            out["pct_change"] = pct
            out["pct_display"] = f"{pct:.1f}%"
    return out


def _primary_spec_ids() -> set[str]:
    return {"main", "h2_m1_period", "d1_m1_total_association"}


def build_main_focal(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["spec_id"].isin(_primary_spec_ids())].copy()
    rows = []
    for mid in MODULE_ORDER:
        focal = FOCAL_BY_MODEL.get(mid, ())
        for term in focal:
            hit = sub[(sub["model_id"] == mid) & (sub["term"] == term)]
            if hit.empty:
                continue
            rows.append(enrich_row(hit.iloc[0]))
    out = pd.DataFrame(rows)
    col_order = [
        "module",
        "model_id",
        "term",
        "dv_zh",
        "sample_zh",
        "predictor_zh",
        "coef",
        "se",
        "pvalue",
        "p_fmt",
        "stars",
        "coef_display",
        "or_display",
        "pct_display",
        "n",
        "estimator",
        "formula_zh",
    ]
    return out[[c for c in col_order if c in out.columns]]


def build_main_full(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["spec_id"].isin(_primary_spec_ids())].copy()
    rows = [enrich_row(sub.iloc[i]) for i in range(len(sub))]
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MODULE_ORDER)}
    out["_ord"] = out["model_id"].map(order)
    out = out.sort_values(["_ord", "term"]).drop(columns="_ord")
    return out


def build_appendix_without_t2(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["spec_id"] == "without_t2_control")
        | (df["spec_id"].astype(str).str.endswith("_without_t2"))
    ].copy()
    rows = []
    for mid in MODULE_ORDER:
        if mid == "h2_indirect_period":
            continue
        for term in FOCAL_BY_MODEL.get(mid, ()):
            hit = sub[(sub["model_id"] == mid) & (sub["term"] == term)]
            if hit.empty:
                continue
            rows.append(enrich_row(hit.iloc[0]))
    return pd.DataFrame(rows)


def build_robustness_focal(df: pd.DataFrame) -> pd.DataFrame:
    skip = _primary_spec_ids() | {"without_t2_control"}
    sub = df[~df["spec_id"].isin(skip)].copy()
    sub = sub[~sub["spec_id"].astype(str).str.endswith("_without_t2")]
    rows = []
    for mid in MODULE_ORDER:
        focal = FOCAL_BY_MODEL.get(mid, ())
        for spec in sorted(sub["spec_id"].unique()):
            for term in focal:
                hit = sub[
                    (sub["model_id"] == mid) & (sub["spec_id"] == spec) & (sub["term"] == term)
                ]
                if hit.empty:
                    continue
                rows.append(enrich_row(hit.iloc[0]))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {m: i for i, m in enumerate(MODULE_ORDER)}
    out["_ord"] = out["model_id"].map(order)
    return out.sort_values(["_ord", "spec_id", "term"]).drop(columns="_ord")


def build_robustness_wide(df: pd.DataFrame) -> pd.DataFrame:
    focal_df = build_robustness_focal(df)
    if focal_df.empty:
        return focal_df
    key = focal_df["module"] + " | " + focal_df["predictor_zh"]
    wide = focal_df.pivot_table(
        index=key,
        columns="spec_label_zh",
        values="coef_display",
        aggfunc="first",
    )
    wide = wide.reset_index().rename(columns={"index": "model_predictor"})
    return wide


def build_main_vs_appendix_compare(df: pd.DataFrame) -> pd.DataFrame:
    main = build_main_focal(df)
    wot = build_appendix_without_t2(df)
    if main.empty or wot.empty:
        return pd.DataFrame()
    m = main[
        ["module", "model_id", "term", "predictor_zh", "coef", "se", "pvalue", "coef_display", "n"]
    ].rename(
        columns={
            "coef": "coef_main",
            "se": "se_main",
            "pvalue": "p_main",
            "coef_display": "display_main",
            "n": "n_main",
        }
    )
    w = wot[["model_id", "term", "coef", "se", "pvalue", "coef_display", "n"]].rename(
        columns={
            "coef": "coef_no_t2",
            "se": "se_no_t2",
            "pvalue": "p_no_t2",
            "coef_display": "display_no_t2",
            "n": "n_no_t2",
        }
    )
    merged = m.merge(w, on=["model_id", "term"], how="left")
    merged["delta_coef"] = merged["coef_main"] - merged["coef_no_t2"]
    merged["n_main"] = as_int64(merged["n_main"])
    merged["n_no_t2"] = as_int64(merged["n_no_t2"])
    merged["n_main_fmt"] = fmt_n_display(merged["n_main"])
    merged["n_no_t2_fmt"] = fmt_n_display(merged["n_no_t2"])
    col_order = [
        "module",
        "model_id",
        "term",
        "predictor_zh",
        "coef_main",
        "se_main",
        "p_main",
        "display_main",
        "n_main",
        "n_main_fmt",
        "coef_no_t2",
        "se_no_t2",
        "p_no_t2",
        "display_no_t2",
        "n_no_t2",
        "n_no_t2_fmt",
        "delta_coef",
    ]
    return merged[[c for c in col_order if c in merged.columns]]


def write_markdown_main(path: Path, focal: pd.DataFrame) -> None:
    lines = [
        "# 主回归结果（focal predictors）",
        "",
        "注：括号内为按账号聚类的标准误。\\* *p*<0.05, \\*\\* *p*<0.01, \\*\\*\\* *p*<0.001。",
        "Logit 模型另报告 odds ratio（OR）。",
        "",
        "| 模块 | 因变量 | 样本 | 预测变量 | 系数 (SE) | *p* | OR / %Δ | *N* |",
        "|------|--------|------|----------|-----------|-----|---------|-----|",
    ]
    for _, r in focal.iterrows():
        extra = ""
        for col in ("or_display", "pct_display"):
            val = r.get(col, "")
            if pd.notna(val) and str(val).strip():
                extra = str(val)
                break
        lines.append(
            f"| {r['module']} | {r['dv_zh']} | {r['sample_zh']} | {r['predictor_zh']} | "
            f"{r['coef_display']} | {r['p_fmt']} | {extra} | {r['n']:,} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=IN_ALL)
    args = p.parse_args()

    df = load_summary(args.input.expanduser().resolve())
    # Keep only formal H1 / H2 / D1 model ids if a merged dump is broader.
    formal_ids = set(MODULE_ORDER)
    df = df[df["model_id"].isin(formal_ids)].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    built = utc_now_iso()

    focal = build_main_focal(df)
    full = build_main_full(df)
    wot = build_appendix_without_t2(df)
    rob = build_robustness_focal(df)
    wide = build_robustness_wide(df)
    compare = build_main_vs_appendix_compare(df)

    outputs = {
        "table_4_main_regression_focal.csv": focal,
        "table_4_main_regression_full.csv": full,
        "appendix_A1_robustness_focal.csv": rob,
        "appendix_A2_without_t2_control.csv": wot,
        "appendix_A3_robustness_wide.csv": wide,
        "appendix_A4_main_vs_without_t2.csv": compare,
    }
    for name, frame in outputs.items():
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["built_at"] = built
        out_path = OUT_DIR / name
        frame.to_csv(out_path, index=False, encoding="utf-8-sig")

    if not focal.empty:
        write_markdown_main(OUT_DIR / "table_4_main_regression.md", focal)

    print(f"regression tables -> {OUT_DIR}")
    for name, frame in outputs.items():
        if frame is not None and not frame.empty:
            print(f"  {name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
