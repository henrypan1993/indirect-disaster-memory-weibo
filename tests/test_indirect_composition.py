"""Tests for exploratory peripheral Indirect/Mixed composition analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "explore_comp",
    SCRIPTS / "11_explore_indirect_composition.py",
)
assert _spec is not None and _spec.loader is not None
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def test_emotion_polarity_four_way_mapping():
    s = pd.Series(
        [
            "Anger",
            "Hope",
            "Neutral-Informational",
            "Mixed-Ambivalent",
            "Other-Unclear",
            "Sadness-Grief",
        ]
    )
    out = common.emotion_polarity_series(s)
    assert out.iloc[0] == "negative"
    assert out.iloc[1] == "positive"
    assert out.iloc[2] == "neutral"
    assert out.iloc[3] == "mixed"
    assert pd.isna(out.iloc[4])
    assert out.iloc[5] == "negative"
    assert common.EMOTION_POLARITY_MAP["Mixed-Ambivalent"] == "mixed"
    assert "Mixed-Ambivalent" not in {
        k for k, v in common.EMOTION_POLARITY_MAP.items() if v == "negative"
    }


def test_select_sample_peripheral_indirect_only():
    df = pd.DataFrame(
        {
            "include_main": [1, 1, 1, 1],
            "peripheral": [1, 1, 0, 1],
            "indirect_clean": [1.0, 0.0, 1.0, 1.0],
            "label_narrative_clean": [
                "Memory-Reactivation",
                "Trauma-Help-Loss",
                "Memory-Reactivation",
                "Restoration",
            ],
            "label_emotion_clean": [
                "Anger",
                "Hope",
                "Hope",
                "Mixed-Ambivalent",
            ],
            "t2": [0, 0, 1, 1],
        }
    )
    sample = ex.select_peripheral_indirect(df)
    # rows 0 and 3 only (peripheral & indirect)
    assert len(sample) == 2
    assert set(sample["label_narrative_clean"]) == {"Memory-Reactivation", "Restoration"}
    mem = sample.loc[sample["label_narrative_clean"] == "Memory-Reactivation"]
    rest = sample.loc[sample["label_narrative_clean"] == "Restoration"]
    mixed = sample.loc[sample["label_emotion_clean"] == "Mixed-Ambivalent"]
    assert int(mem["memory_reactivation"].iloc[0]) == 1
    assert int(rest["memory_reactivation"].iloc[0]) == 0
    assert mixed["emotion_polarity"].iloc[0] == "mixed"


def test_cramers_v_and_chi2_report():
    df = pd.DataFrame(
        {
            "memory_reactivation": [0, 0, 0, 1, 1, 1, 1, 1],
            "t2": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    v = ex.cramers_v(4.0, 8, 2, 2)
    assert np.isclose(v, np.sqrt(4.0 / (8 * 1)))
    rep = ex.chi2_report(df, "memory_reactivation")
    assert "cramers_v" in rep
    assert np.isfinite(rep["cramers_v"])
    assert rep["n"] == 8


def test_m2_formulas_structure():
    m2a = ex._m2a_spec()
    m2b = ex._m2b_spec()
    fa = common.build_formula(m2a)
    fb = common.build_formula(m2b)
    assert "memory_reactivation" in fa
    assert "label_narrative_clean" not in fa
    assert "emotion_polarity" in fa
    assert "label_narrative_clean" in fb
    assert "memory_reactivation" not in fb
    form_c = (
        "log_engagement ~ memory_reactivation + t2 + memory_reactivation:t2 + "
        + " + ".join(common.PERIPHERAL_SAMPLE_CONTROLS)
        + " + C(topic_id, Treatment(reference='0'))"
    )
    assert "memory_reactivation:t2" in form_c
    assert "emotion_polarity" not in form_c
