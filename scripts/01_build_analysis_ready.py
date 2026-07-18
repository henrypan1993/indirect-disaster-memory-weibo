"""Build base analysis-ready table with derived controls and sample flags."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from typing import Any

import numpy as np
import pandas as pd
from common import (
    PATH_ANALYSIS_BASE,
    PATH_INPUT_CSV,
    PROJECT_ROOT,
    analysis_text_series,
    count_hashtags,
    narrative_trauma_clean,
    utc_now_iso,
    write_json_report,
)


def build_analysis_ready(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    text = analysis_text_series(out)
    out["text_length"] = text.map(lambda t: len(str(t).strip()) if pd.notna(t) else 0)
    out["hashtag_count"] = text.map(count_hashtags)

    if "created_at_dt" in out.columns:
        ts = pd.to_datetime(out["created_at_dt"], errors="coerce", utc=True)
        ts = ts.dt.tz_convert("Asia/Shanghai")
        out["post_hour_cst"] = ts.dt.hour
        out["post_date_cst"] = ts.dt.date.astype(str)
        out["weekday_cst"] = ts.dt.dayofweek
    else:
        out["post_hour_cst"] = np.nan
        out["post_date_cst"] = pd.NA
        out["weekday_cst"] = np.nan

    for c in ("likes_count", "comments_count", "reposts_count"):
        out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0)
    raw_eng = np.log1p(out["likes_count"] + out["comments_count"] + out["reposts_count"])
    out["log_engagement"] = raw_eng
    if "engagement" in out.columns:
        existing = pd.to_numeric(out["engagement"], errors="coerce")
        mismatch = (existing - raw_eng).abs() > 1e-6
        if mismatch.any():
            n_bad = int(mismatch.sum())
            msg = f"engagement mismatch vs log1p sum: {n_bad} rows"
            raise ValueError(msg)
    out["log_likes"] = np.log1p(out["likes_count"])
    out["log_comments"] = np.log1p(out["comments_count"])
    out["log_reposts"] = np.log1p(out["reposts_count"])

    out["robust_no_review"] = ~out["needs_manual_review"].fillna(True)

    include = out["include_main"] == 1
    out["narrative_trauma_clean"] = narrative_trauma_clean(
        out["label_narrative_clean"].astype("string")
    )
    out["disaster_impact_related"] = out["narrative_trauma_clean"]
    out["model_sample_h1_indirect"] = include & out["indirect_clean"].notna()
    out["model_sample_h2"] = include & (out["peripheral"] == 1)
    out["model_sample_h4_indirect"] = (
        include & (out["peripheral"] == 1) & (out["narrative_trauma_clean"] == 1)
    )

    stats: dict[str, Any] = {
        "n_rows": len(out),
        "n_include_main": int(include.sum()),
        "n_model_sample_h1_indirect": int(out["model_sample_h1_indirect"].sum()),
        "n_model_sample_h2": int(out["model_sample_h2"].sum()),
        "n_model_sample_h4_indirect": int(out["model_sample_h4_indirect"].sum()),
        "n_robust_no_review": int(out["robust_no_review"].sum()),
        "indirect_clean_na": int(out["indirect_clean"].isna().sum()),
        "entropy_norm_missing": True,
    }
    return out, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_INPUT_CSV)
    p.add_argument("--output", type=Path, default=PATH_ANALYSIS_BASE)
    p.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "analysis_ready_report.json",
    )
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str})
    ready, stats = build_analysis_ready(df)

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    ready.to_csv(out, index=False, encoding="utf-8-sig")

    report = {"built_at": utc_now_iso(), "input_csv": str(inp), "output_csv": str(out), **stats}
    write_json_report(args.report, report)
    print(f"analysis_ready_base: {len(ready)} rows -> {out}")


if __name__ == "__main__":
    main()
