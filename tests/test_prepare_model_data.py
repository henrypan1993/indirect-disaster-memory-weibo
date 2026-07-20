"""Smoke tests for prepare_model_data helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "prepare_model_data",
    SCRIPTS / "03_prepare_model_data.py",
)
assert _spec is not None and _spec.loader is not None
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def test_prepare_model_data_flags_and_p90():
    df = pd.DataFrame(
        {
            "mid": ["1", "2", "3", "4"],
            "account_id": ["a", "b", "c", "d"],
            "include_main": [1, 1, 1, 1],
            "t2": [0, 0, 1, 1],
            "verified": [0, 1, 0, 0],
            "followers_count": [100, 1_000_000, 200, 50],
            "log_followers": [4.6, 13.8, 5.3, 3.9],
            "indirect_clean": [0.0, 1.0, 1.0, None],
            "label_narrative_clean": [
                "Trauma-Help-Loss",
                "Restoration",
                "Memory-Reactivation",
                "Trauma-Help-Loss",
            ],
            "label_emotion_clean": [
                "Neutral-Informational",
                "Hope",
                "Sadness-Grief",
                "Fear-Anxiety",
            ],
            "label_expression_clean": ["Direct", "Indirect-Mixed", "Indirect-Mixed", "Unclear"],
            "topic_id": [0, 1, 0, 2],
            "post_hour_cst": [10, 11, 12, 13],
            "post_date_cst": [
                "2021-07-18",
                "2021-07-19",
                "2025-08-04",
                "2025-08-05",
            ],
            "hashtag_count": [0, 1, 0, 0],
            "text_length": [10, 20, 30, 5],
            "log_engagement": [1.0, 2.0, 1.5, 0.1],
            "log_likes": [0.5, 1.0, 0.8, 0.0],
            "log_comments": [0.1, 0.2, 0.1, 0.0],
            "log_reposts": [0.0, 0.1, 0.0, 0.0],
            "likes_count": [1, 2, 1, 0],
            "comments_count": [0, 1, 0, 0],
            "reposts_count": [0, 0, 0, 0],
            "high_clarity": [True, True, True, False],
            "needs_manual_review": [False, False, False, True],
        }
    )
    frozen, report = prep.prepare_model_data(df)
    assert "sample_h1_eligible" in frozen.columns
    assert "relative_window_day" in frozen.columns
    assert "disaster_impact_related" in frozen.columns
    assert "entropy_norm" not in frozen.columns
    assert report["n_sample_d1_eligible"] == 3  # three non-null indirect_clean
    assert frozen.loc[0, "relative_window_day"] == 0.0
    assert frozen.loc[2, "relative_window_day"] == 0.0
