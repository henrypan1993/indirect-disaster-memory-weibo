"""Before/after audit for control-set fix (peripheral models drop verified)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT

OUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"
ARCHIVE_DIR = OUT_MODELS / "_archive_pre_control_fix"

AUDIT_MODELS: dict[str, list[str]] = {
    "h1_engagement_indirect": ["indirect_clean"],
    "h2a_entropy_reactivation": ["t2"],
    "h2b_indirect_reactivation": ["t2"],
    "e2_entropy_increment": ["indirect_clean", "entropy_norm"],
}


def parse_condition_number(txt_path: Path) -> float | None:
    if not txt_path.is_file():
        return None
    text = txt_path.read_text(encoding="utf-8")
    m = re.search(r"Cond\. No\.\s+([\d.eE+-]+)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"condition_number=([\d.eE+-]+)", text)
    if m:
        return float(m.group(1))
    return None


def parse_covariance(txt_path: Path) -> str | None:
    if not txt_path.is_file():
        return None
    text = txt_path.read_text(encoding="utf-8")
    m = re.search(r"Covariance Type:\s+(\S+)", text)
    return m.group(1) if m else None


def focal_from_summary(
    df: pd.DataFrame, model_id: str, term: str
) -> tuple[float | None, float | None]:
    sub = df.loc[(df["model_id"] == model_id) & (df["spec_id"] == "main") & (df["term"] == term)]
    if sub.empty:
        return None, None
    row = sub.iloc[0]
    return float(row["coef"]), float(row["se"])


def meta_for(model_id: str) -> dict | None:
    path = OUT_MODELS / f"{model_id}_main_meta.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archive",
        type=Path,
        default=ARCHIVE_DIR,
        help="Directory with pre-fix main_models_summary.csv and *.txt",
    )
    args = p.parse_args()

    archive = args.archive.expanduser().resolve()
    before_csv = archive / "main_models_summary.csv"
    after_csv = OUT_MODELS / "main_models_summary.csv"

    if not before_csv.is_file():
        raise SystemExit(f"archive not found: {before_csv}")
    if not after_csv.is_file():
        raise SystemExit(f"current summary not found: {after_csv} (run 04 first)")

    before = pd.read_csv(before_csv)
    after = pd.read_csv(after_csv)

    rows: list[dict] = []
    for model_id, focal_terms in AUDIT_MODELS.items():
        meta = meta_for(model_id) or {}
        diag = meta.get("diagnostics") or {}
        txt_after = OUT_MODELS / f"{model_id}.txt"
        txt_before = archive / f"{model_id}.txt"

        for term in focal_terms:
            cb, sb = focal_from_summary(before, model_id, term)
            ca, sa = focal_from_summary(after, model_id, term)
            rows.append(
                {
                    "model_id": model_id,
                    "removed_variable": "verified",
                    "n": diag.get("n_obs"),
                    "n_accounts": diag.get("n_accounts"),
                    "covariance": meta.get("covariance_estimator")
                    or parse_covariance(txt_after),
                    "focal_term": term,
                    "coef_before": cb,
                    "coef_after": ca,
                    "se_before": sb,
                    "se_after": sa,
                    "condition_number_before": parse_condition_number(txt_before),
                    "condition_number_after": diag.get("condition_number")
                    or parse_condition_number(txt_after),
                    "notes": "",
                }
            )

    out_df = pd.DataFrame(rows)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = OUT_REPORTS / "model_spec_audit_before_after.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"audit -> {out_path} ({len(out_df)} rows)")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
