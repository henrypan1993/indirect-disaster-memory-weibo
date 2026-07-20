"""Robustness: H1 components, H2-M2, D1-M2, peripheral thresholds."""

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
    PROJECT_ROOT,
    composition_or_conditional_specs,
    run_model_spec,
    spec_by_spec_id,
    utc_now_iso,
)

OUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"

PERIPHERAL_PERCENTILES = (80, 90, 95)


def alt_peripheral_frame(df: pd.DataFrame, pct: int) -> tuple[pd.DataFrame, float, int]:
    """Recompute peripheral at pXX on include_main; refresh eligible flags."""
    out = df.copy()
    followers = pd.to_numeric(out["followers_count"], errors="coerce")
    eligible = out["include_main"] == 1
    threshold = float(np.nanpercentile(followers[eligible], pct))
    verified = pd.to_numeric(out["verified"], errors="coerce").fillna(0).astype(int)
    peripheral_alt = ((verified == 0) & (followers < threshold)).astype(int)
    out["peripheral"] = peripheral_alt
    expr_ok = out["indirect_clean"].notna()
    impact = (
        out["disaster_impact_related"]
        if "disaster_impact_related" in out
        else out.get("narrative_trauma_clean", 0)
    )
    out["sample_h1_eligible"] = eligible & (peripheral_alt == 1) & expr_ok
    out["sample_h2_eligible"] = eligible & (peripheral_alt == 1) & (impact == 1) & expr_ok
    out["sample_d1_eligible"] = eligible & expr_ok
    return out, threshold, int(peripheral_alt[eligible].sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_MODEL_DATA_FINAL)
    p.add_argument(
        "--merge",
        action="store_true",
        help="Write robustness_models_summary.csv (always on in this script).",
    )
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str})
    if "verified" in df.columns:
        df["verified"] = df["verified"].fillna(0).astype(int)

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    built_at = utc_now_iso()

    # H1 engagement components
    h1 = spec_by_spec_id("main")
    for ycol, sid in (
        ("log_likes", "engagement_likes"),
        ("log_comments", "engagement_comments"),
        ("log_reposts", "engagement_reposts"),
    ):
        print(f"H1 component {sid} ...")
        part, summary, _ = run_model_spec(df, h1, sid, y_col=ycol, meta_dir=OUT_MODELS)
        rows.extend(part)
        (OUT_MODELS / f"h1_engagement_indirect_{sid}.txt").write_text(summary, encoding="utf-8")

    # H2-M2 and D1-M2
    for spec in composition_or_conditional_specs():
        sid = spec.default_spec_id
        print(f"{spec.formal_label} {sid} ...")
        part, summary, _ = run_model_spec(df, spec, sid, meta_dir=OUT_MODELS)
        rows.extend(part)
        (OUT_MODELS / f"{spec.model_id}_{sid}.txt").write_text(summary, encoding="utf-8")

    # Peripheral thresholds for H1 and D1-M1
    d1 = spec_by_spec_id("d1_m1_total_association")
    for pct in PERIPHERAL_PERCENTILES:
        alt_df, threshold, n_periph = alt_peripheral_frame(df, pct)
        sid = f"peripheral_p{pct}"
        notes = (
            f"peripheral = verified==0 & followers<p{pct} ({threshold:.0f}); "
            f"n_peripheral={n_periph}"
        )
        print(f"{sid} ...")
        for spec in (h1, d1):
            part, summary, _ = run_model_spec(
                alt_df,
                spec,
                sid,
                notes=notes,
                meta_dir=OUT_MODELS,
            )
            rows.extend(part)
            (OUT_MODELS / f"{spec.model_id}_{sid}.txt").write_text(summary, encoding="utf-8")

    out_df = pd.DataFrame(rows)
    out_df["built_at"] = built_at
    out_path = OUT_MODELS / "robustness_models_summary.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"robustness -> {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
