"""Validate canonical input table before modeling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from typing import Any

import pandas as pd
from common import (
    EXPECTED_ROW_COUNT,
    PATH_INPUT_CSV,
    PROJECT_ROOT,
    REQUIRED_INPUT_COLS,
    utc_now_iso,
    write_json_report,
)


def run_checks(df: pd.DataFrame, *, expected_rows: int = EXPECTED_ROW_COUNT) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    n = len(df)
    checks["row_count"] = {
        "expected": expected_rows,
        "actual": n,
        "passed": n == expected_rows,
    }

    mid_unique = df["mid"].is_unique
    checks["mid_unique"] = {"passed": bool(mid_unique)}

    for col in REQUIRED_INPUT_COLS:
        present = col in df.columns
        checks[f"column_{col}"] = {"passed": present}

    all_passed = all(c.get("passed", False) for c in checks.values())
    return {
        "built_at": utc_now_iso(),
        "input_csv": str(PATH_INPUT_CSV),
        "passed": all_passed,
        "checks": checks,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_INPUT_CSV)
    p.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "input_check_report.json",
    )
    p.add_argument(
        "--expected-rows",
        type=int,
        default=EXPECTED_ROW_COUNT,
        help="Override for tests (default 17143).",
    )
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        report = {
            "built_at": utc_now_iso(),
            "input_csv": str(inp),
            "passed": False,
            "error": "input file not found",
        }
        write_json_report(args.report, report)
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str})
    report = run_checks(df, expected_rows=args.expected_rows)
    report["input_csv"] = str(inp)
    write_json_report(args.report, report)

    if not report["passed"]:
        raise SystemExit("input check failed; see report")

    print(f"input check passed ({len(df)} rows) -> {args.report}")


if __name__ == "__main__":
    main()
