"""Tests for stacked engagement-asymmetry Wald tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "eng_asym",
    SCRIPTS / "13_test_engagement_asymmetry.py",
)
assert _spec is not None and _spec.loader is not None
asym = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asym)


def _toy_core(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    indirect = rng.integers(0, 2, size=n).astype(float)
    # Asymmetry: likes/comments respond to indirect; reposts do not
    noise = rng.normal(0, 0.3, size=n)
    likes = 0.4 * indirect + noise + rng.normal(0, 0.1, size=n)
    comments = 0.25 * indirect + noise + rng.normal(0, 0.1, size=n)
    reposts = 0.02 * indirect + rng.normal(0, 0.2, size=n)
    return pd.DataFrame(
        {
            "account_id": [f"a{i % 12}" for i in range(n)],
            "mid": [f"m{i}" for i in range(n)],
            "indirect_clean": indirect,
            "t2": rng.integers(0, 2, size=n).astype(float),
            "log_followers": rng.normal(3, 1, size=n),
            "post_hour_cst": rng.integers(0, 24, size=n).astype(float),
            "hashtag_count": rng.integers(0, 5, size=n).astype(float),
            "text_length": rng.integers(20, 200, size=n).astype(float),
            "topic_id": [str(i % 3) for i in range(n)],
            "log_likes": likes,
            "log_comments": comments,
            "log_reposts": reposts,
        }
    )


def test_stack_triples_rows_and_labels():
    core = _toy_core(10)
    stacked = asym.stack_engagement_outcomes(core)
    assert len(stacked) == 30
    assert set(stacked["outcome"]) == {"likes", "comments", "reposts"}
    assert stacked["y"].notna().all()


def test_formula_uses_reposts_reference_and_focal():
    f = asym.stacked_formula()
    assert f.startswith("y ~")
    assert "reference='reposts'" in f
    assert "indirect_clean" in f
    assert "C(topic_id" in f


def test_stacked_matches_separate_and_rejects_equality():
    core = _toy_core(80, seed=7)
    separate = asym.component_coefs_separate(core)
    stacked_df = asym.stack_engagement_outcomes(core)
    fit, n = asym.fit_ols_cluster(stacked_df, asym.stacked_formula())
    assert n == len(core) * 3
    stacked = asym.component_coefs_stacked(fit, n_posts=len(core))
    merged = separate.merge(stacked, on="outcome", suffixes=("_sep", "_stk"))
    assert float((merged["coef_sep"] - merged["coef_stk"]).abs().max()) < 1e-8

    tests = asym.run_asymmetry_tests(fit)
    assert set(tests["contrast"]) == {
        "joint_equality",
        "likes_vs_reposts",
        "comments_vs_reposts",
        "likes_vs_comments",
    }
    joint = tests.loc[tests["contrast"] == "joint_equality"].iloc[0]
    likes = tests.loc[tests["contrast"] == "likes_vs_reposts"].iloc[0]
    assert int(joint["df_num"]) == 2
    assert int(likes["df_num"]) == 1
    assert int(joint["df_denom"]) == int(likes["df_denom"])
    assert str(joint["df_display"]).startswith("F(2,")
    joint_p = float(joint["pvalue"])
    likes_p = float(likes["pvalue"])
    # Toy DGP has clear likes vs reposts gap
    assert likes_p < 0.05
    assert joint_p < 0.05
