from importlib import import_module

import numpy as np
import pandas as pd
from conftest import FIXTURES

build_mod = import_module("01_build_analysis_ready")


def test_build_analysis_ready_derived_columns():
    df = pd.read_csv(FIXTURES / "mini_labels.csv", dtype={"mid": str})
    ready, stats = build_mod.build_analysis_ready(df)

    assert "text_length" in ready.columns
    assert "hashtag_count" in ready.columns
    assert "log_engagement" in ready.columns
    assert ready.loc[ready["mid"] == "m001", "hashtag_count"].iloc[0] == 1

    row = ready.loc[ready["mid"] == "m001"].iloc[0]
    expected_log = np.log1p(10 + 2 + 1)
    assert abs(row["log_engagement"] - expected_log) < 1e-6

    assert ready["model_sample_h2"].sum() == int(
        ((ready["include_main"] == 1) & (ready["peripheral"] == 1)).sum()
    )
    assert stats["n_rows"] == len(df)
