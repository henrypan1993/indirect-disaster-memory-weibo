from importlib import import_module

import pandas as pd
from common import REQUIRED_INPUT_COLS
from conftest import FIXTURES

check_mod = import_module("00_check_input")


def test_run_checks_passes_on_fixture():
    path = FIXTURES / "mini_labels.csv"
    df = pd.read_csv(path, dtype={"mid": str})
    report = check_mod.run_checks(df, expected_rows=len(df))
    assert report["passed"] is True
    for col in REQUIRED_INPUT_COLS:
        assert report["checks"][f"column_{col}"]["passed"]


def test_mid_unique_fails():
    df = pd.read_csv(FIXTURES / "mini_labels.csv", dtype={"mid": str})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    report = check_mod.run_checks(df, expected_rows=len(df))
    assert report["checks"]["mid_unique"]["passed"] is False
    assert report["passed"] is False
