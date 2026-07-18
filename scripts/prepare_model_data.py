"""Freeze model_data_final.csv for H1 / H2 / D1 from analysis_ready_with_topics.

Fixes peripheral (p90 on include_main), relative_window_day, and eligible sample flags.
Does not redistribute microdata; output stays under data/processed/ (gitignored).
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
    EXPECTED_INCLUDE_MAIN,
    PATH_ANALYSIS_TOPICS,
    PATH_MODEL_DATA_FINAL,
    PROJECT_ROOT,
    disaster_impact_related,
    relative_window_day_series,
    utc_now_iso,
    write_json_report,
)

# Columns kept for modeling (no raw analysis text).
MODEL_DATA_COLS = (
    "mid",
    "account_id",
    "include_main",
    "t2",
    "period",
    "wave",
    "peripheral",
    "verified",
    "followers_count",
    "log_followers",
    "indirect_clean",
    "label_expression_clean",
    "label_narrative_clean",
    "label_emotion_clean",
    "disaster_impact_related",
    "narrative_trauma_clean",
    "topic_id",
    "post_hour_cst",
    "post_date_cst",
    "relative_window_day",
    "hashtag_count",
    "text_length",
    "log_engagement",
    "log_likes",
    "log_comments",
    "log_reposts",
    "likes_count",
    "comments_count",
    "reposts_count",
    "high_clarity",
    "needs_manual_review",
    "sample_h1_eligible",
    "sample_h2_eligible",
    "sample_d1_eligible",
    "p90_followers",
)


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    include = out["include_main"] == 1
    n_include = int(include.sum())
    if n_include != EXPECTED_INCLUDE_MAIN:
        # Soft warning in report; still proceed so local subsets can be tested.
        include_note = f"include_main sum={n_include} (expected {EXPECTED_INCLUDE_MAIN})"
    else:
        include_note = f"include_main sum={n_include} (== Gate-passed/Core N)"

    followers = pd.to_numeric(out["followers_count"], errors="coerce")
    verified = pd.to_numeric(out["verified"], errors="coerce").fillna(0).astype(int)
    p90 = float(np.nanpercentile(followers[include], 90))
    peripheral = ((verified == 0) & (followers < p90)).astype(int)
    out["peripheral"] = peripheral
    out["p90_followers"] = p90
    out["verified"] = verified

    if "log_followers" not in out.columns or out["log_followers"].isna().all():
        out["log_followers"] = np.log1p(followers)

    # Dates for relative_window_day
    if "post_date_cst" not in out.columns or out["post_date_cst"].isna().all():
        if "created_at_dt" in out.columns:
            ts = pd.to_datetime(out["created_at_dt"], errors="coerce", utc=True)
            ts = ts.dt.tz_convert("Asia/Shanghai")
            out["post_date_cst"] = ts.dt.date.astype(str)
        else:
            out["post_date_cst"] = pd.NA

    out["relative_window_day"] = relative_window_day_series(out["post_date_cst"], out["t2"])

    narr = out["label_narrative_clean"].astype("string")
    out["disaster_impact_related"] = disaster_impact_related(narr)
    out["narrative_trauma_clean"] = out["disaster_impact_related"]

    expr_ok = out["indirect_clean"].notna()
    out["sample_h1_eligible"] = include & (out["peripheral"] == 1) & expr_ok
    out["sample_h2_eligible"] = (
        include & (out["peripheral"] == 1) & (out["disaster_impact_related"] == 1) & expr_ok
    )
    out["sample_d1_eligible"] = include & expr_ok

    # Keep whitelist columns that exist
    keep = [c for c in MODEL_DATA_COLS if c in out.columns]
    frozen = out[keep].copy()

    report: dict[str, Any] = {
        "built_at": utc_now_iso(),
        "include_main_note": include_note,
        "n_rows": len(frozen),
        "n_include_main": n_include,
        "p90_followers": p90,
        "n_peripheral_include": int(peripheral[include].sum()),
        "n_sample_h1_eligible": int(out["sample_h1_eligible"].sum()),
        "n_sample_h2_eligible": int(out["sample_h2_eligible"].sum()),
        "n_sample_d1_eligible": int(out["sample_d1_eligible"].sum()),
        "n_relative_window_day_na": int(out["relative_window_day"].isna().sum()),
        "columns": keep,
    }
    return frozen, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_ANALYSIS_TOPICS)
    p.add_argument("--output", type=Path, default=PATH_MODEL_DATA_FINAL)
    p.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "model_data_final_report.json",
    )
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(
            f"input not found: {inp}\n"
            "Run scripts/02_build_topic_entropy.py --from-entropy ... first."
        )

    df = pd.read_csv(inp, dtype={"mid": str})
    if "topic_id" not in df.columns:
        raise SystemExit("input missing topic_id; use analysis_ready_with_topics.csv")

    frozen, report = prepare_model_data(df)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_csv(out, index=False, encoding="utf-8-sig")
    report["input_csv"] = str(inp)
    report["output_csv"] = str(out)
    write_json_report(args.report, report)
    print(
        f"model_data_final: {len(frozen)} rows -> {out} "
        f"(h1={report['n_sample_h1_eligible']}, "
        f"h2={report['n_sample_h2_eligible']}, "
        f"d1={report['n_sample_d1_eligible']})"
    )


if __name__ == "__main__":
    main()
