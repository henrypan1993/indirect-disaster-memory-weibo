"""Fit H1/H2 main models and E1/E2 boundary checks (with T2 controls)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    PATH_ANALYSIS_ENTROPY,
    PROJECT_ROOT,
    appendix_without_t2_specs,
    main_model_specs,
    run_model_spec,
    utc_now_iso,
)

OUT_MODELS = PROJECT_ROOT / "outputs" / "models"


def run_all_specs(
    df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    main_rows: list[dict[str, Any]] = []
    appendix_rows: list[dict[str, Any]] = []
    texts: dict[str, str] = {}

    for spec in main_model_specs():
        rows, summary, _ = run_model_spec(df, spec, "main", meta_dir=OUT_MODELS)
        main_rows.extend(rows)
        texts[f"{spec.model_id}.txt"] = summary

    for spec in appendix_without_t2_specs():
        rows, summary, _ = run_model_spec(
            df, spec, "without_t2_control", meta_dir=OUT_MODELS
        )
        appendix_rows.extend(rows)
        texts[f"{spec.model_id}_without_t2.txt"] = summary

    return main_rows, appendix_rows, texts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_ANALYSIS_ENTROPY)
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str})
    if "verified" in df.columns:
        df["verified"] = df["verified"].fillna(0).astype(int)

    main_rows, appendix_rows, texts = run_all_specs(df)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)

    built_at = utc_now_iso()
    main_df = pd.DataFrame(main_rows)
    main_df["built_at"] = built_at
    main_path = OUT_MODELS / "main_models_summary.csv"
    main_df.to_csv(main_path, index=False, encoding="utf-8-sig")

    appendix_df = pd.DataFrame(appendix_rows)
    appendix_df["built_at"] = built_at
    appendix_path = OUT_MODELS / "appendix_without_t2.csv"
    appendix_df.to_csv(appendix_path, index=False, encoding="utf-8-sig")

    for name, body in texts.items():
        (OUT_MODELS / name).write_text(body, encoding="utf-8")

    print(
        f"main models -> {main_path} ({len(main_rows)} rows); "
        f"appendix -> {appendix_path} ({len(appendix_rows)} rows)"
    )


if __name__ == "__main__":
    main()
