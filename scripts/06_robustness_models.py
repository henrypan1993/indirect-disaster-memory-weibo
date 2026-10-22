"""Robustness: H1 components, H1/H2-M2, broader H1, D1-M2, peripheral thresholds."""

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
    broader_sample_specs,
    composition_or_conditional_specs,
    run_model_spec,
    spec_by_spec_id,
    utc_now_iso,
)

OUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"
OUT_PUBLIC = PROJECT_ROOT / "outputs" / "public"

PERIPHERAL_PERCENTILES = (80, 90, 95)

# July 2026 archive (outputs/models/all_models_summary.csv). Used only to
# decide whether a refit may be published. The refit itself is computed.
ARCHIVE_Q2_COMPONENTS = {
    "log_likes": (0.10184501680049007, 10806),
    "log_comments": (0.07044548729477235, 10806),
    "log_reposts": (0.007091460132850225, 10806),
}
ARCHIVE_Q2_THRESHOLDS = {
    80: (0.12648850223216826, 10650),
    90: (0.13225046964954723, 10806),
    95: (0.12970903603346817, 10853),
}
COMPONENT_SPEC_IDS = {
    "log_likes": "engagement_likes_broader",
    "log_comments": "engagement_comments_broader",
    "log_reposts": "engagement_reposts_broader",
}


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
    core = eligible & (peripheral_alt == 1) & (impact == 1) & expr_ok
    out["sample_h1_eligible"] = core
    out["sample_h2_eligible"] = core
    out["sample_h1_broader_eligible"] = eligible & (peripheral_alt == 1) & expr_ok
    out["sample_d1_eligible"] = eligible & expr_ok
    return out, threshold, int(peripheral_alt[eligible].sum())


def _focal_indirect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [row for row in rows if row.get("term") == "indirect_clean"]
    if len(matches) != 1:
        msg = f"expected one indirect_clean row, found {len(matches)}"
        raise RuntimeError(msg)
    return matches[0]


def fit_q2_broader_outputs(df: pd.DataFrame) -> bool:
    """Refit manuscript Q2 component and threshold models on the broader sample.

    Writes public tables only when coefficients match the July 2026 archive.
    Does not replace core-sample component or threshold files.
    """
    broader = spec_by_spec_id("h1_broader_sample")
    component_rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for y_col, spec_id in COMPONENT_SPEC_IDS.items():
        part, summary, _ = run_model_spec(df, broader, spec_id, y_col=y_col, meta_dir=OUT_MODELS)
        focal = _focal_indirect(part)
        expected_coef, expected_n = ARCHIVE_Q2_COMPONENTS[y_col]
        if int(focal["n"]) != expected_n or abs(float(focal["coef"]) - expected_coef) > 1e-8:
            mismatches.append(
                f"{spec_id}: coef={focal['coef']} n={focal['n']} "
                f"(archive coef={expected_coef} n={expected_n})"
            )
        component_rows.append(focal)
        (OUT_MODELS / f"h1_engagement_indirect_{spec_id}.txt").write_text(summary, encoding="utf-8")

    threshold_rows: list[dict[str, Any]] = []
    for pct in PERIPHERAL_PERCENTILES:
        alt_df, threshold, n_periph = alt_peripheral_frame(df, pct)
        spec_id = f"peripheral_p{pct}_broader"
        notes = (
            f"broader peripheral = verified==0 & followers<p{pct} ({threshold:.0f}); "
            f"n_peripheral={n_periph}"
        )
        part, summary, _ = run_model_spec(
            alt_df, broader, spec_id, notes=notes, meta_dir=OUT_MODELS
        )
        focal = _focal_indirect(part)
        focal["follower_cut"] = threshold
        expected_coef, expected_n = ARCHIVE_Q2_THRESHOLDS[pct]
        if int(focal["n"]) != expected_n or abs(float(focal["coef"]) - expected_coef) > 1e-8:
            mismatches.append(
                f"{spec_id}: coef={focal['coef']} n={focal['n']} "
                f"(archive coef={expected_coef} n={expected_n})"
            )
        threshold_rows.append(focal)
        (OUT_MODELS / f"h1_engagement_indirect_{spec_id}.txt").write_text(summary, encoding="utf-8")

    if mismatches:
        OUT_REPORTS.mkdir(parents=True, exist_ok=True)
        report = OUT_REPORTS / "q2_broader_recovery_mismatch.txt"
        report.write_text("\n".join(mismatches) + "\n", encoding="utf-8")
        print("Q2 broader refit did not match the July archive. Public tables not written.")
        for line in mismatches:
            print(line)
        return False

    OUT_PUBLIC.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(component_rows)[["spec_id", "term", "coef", "se", "pvalue", "n"]].to_csv(
        OUT_PUBLIC / "table_q2_engagement_components_broader.csv", index=False
    )
    pd.DataFrame(threshold_rows)[
        ["spec_id", "term", "follower_cut", "coef", "se", "pvalue", "n"]
    ].to_csv(OUT_PUBLIC / "table_q2_peripheral_thresholds_broader.csv", index=False)
    print("Q2 broader tables match the July archive and were written to outputs/public/.")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_MODEL_DATA_FINAL)
    p.add_argument(
        "--merge",
        action="store_true",
        help="Write robustness_models_summary.csv (always on in this script).",
    )
    p.add_argument(
        "--q2-broader-only",
        action="store_true",
        help="Refit broader-sample Q2 components and thresholds without rewriting core outputs.",
    )
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str})
    if "verified" in df.columns:
        df["verified"] = df["verified"].fillna(0).astype(int)
    if args.q2_broader_only:
        ok = fit_q2_broader_outputs(df)
        raise SystemExit(0 if ok else 1)

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    built_at = utc_now_iso()

    # H1 engagement components (core disaster-impact × peripheral sample)
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

    # H1-M2, H2-M2, and D1-M2
    for spec in composition_or_conditional_specs():
        sid = spec.default_spec_id
        print(f"{spec.formal_label} {sid} ...")
        part, summary, _ = run_model_spec(df, spec, sid, meta_dir=OUT_MODELS)
        rows.extend(part)
        (OUT_MODELS / f"{spec.model_id}_{sid}.txt").write_text(summary, encoding="utf-8")

    # Broader-sample robustness: all peripheral posts (any narrative)
    for spec in broader_sample_specs():
        sid = spec.default_spec_id
        print(f"{spec.formal_label} {sid} ...")
        part, summary, _ = run_model_spec(df, spec, sid, meta_dir=OUT_MODELS)
        rows.extend(part)
        (OUT_MODELS / f"{spec.model_id}_{sid}.txt").write_text(summary, encoding="utf-8")

    # Peripheral thresholds for core-sample H1 and D1-M1
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
    fit_q2_broader_outputs(df)


if __name__ == "__main__":
    main()
