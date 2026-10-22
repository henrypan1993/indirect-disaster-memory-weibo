"""Tests for qualitative matrix sampling (script 12)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "qual_matrix",
    SCRIPTS / "12_export_qualitative_matrix.py",
)
assert _spec is not None and _spec.loader is not None
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)


def _row(
    i: int,
    *,
    t2: int,
    indirect: int,
    narrative: str,
    residual: float = 0.0,
    likes_r: float = 0.0,
    comments_r: float = 0.0,
    reposts_r: float = 0.0,
    likes: int = 1,
    comments: int = 1,
    reposts: int = 0,
    pred: float = 1.0,
    needs: bool = False,
    core: bool = True,
) -> dict:
    return {
        "mid": str(i),
        "account_id": f"acc{i}",
        "t2": t2,
        "indirect_clean": float(indirect),
        "label_narrative_clean": narrative,
        "label_expression_clean": "Indirect-Mixed" if indirect else "Direct",
        "label_emotion_clean": "Neutral-Informational",
        "needs_manual_review": needs,
        "high_clarity": True,
        "analysis_text_final": f"text-{i}-content",
        "text_hash": f"h{i}",
        "text_length": 40 + i,
        "disaster_impact_related": int(narrative in {"Trauma-Help-Loss", "Memory-Reactivation"}),
        "sample_h1_eligible": int(
            core and narrative in {"Trauma-Help-Loss", "Memory-Reactivation"}
        ),
        "sample_h1_broader_eligible": 1,
        "residual_engagement": residual,
        "likes_resid": likes_r,
        "comments_resid": comments_r,
        "reposts_resid": reposts_r,
        "likes_count": likes,
        "comments_count": comments,
        "reposts_count": reposts,
        "predicted_log_engagement": pred,
        "raw_engagement": likes + comments + reposts,
        "include_main": 1,
        "peripheral": 1,
    }


def _balanced_frame() -> pd.DataFrame:
    rows: list[dict] = []
    i = 0
    for t2 in (0, 1):
        for narr, n, indirect, core in (
            ("Trauma-Help-Loss", 20, 1, True),
            ("Memory-Reactivation", 12, 1, True),
            ("Ordinary-Weather-Info", 12, 1, False),
            ("Other-Unclear", 8, 1, False),
            ("Trauma-Help-Loss", 16, 0, True),
            ("Memory-Reactivation", 8, 0, True),
        ):
            for _ in range(n):
                rows.append(
                    _row(
                        i,
                        t2=t2,
                        indirect=indirect,
                        narrative=narr,
                        core=core,
                    )
                )
                i += 1
    return pd.DataFrame(rows)


def test_semantic_ids_map_paper_hypotheses():
    ids = common.SEMANTIC_ANALYSIS_IDS
    assert ids["period_to_indirect"]["paper_label"] == "H1"
    assert ids["period_to_indirect"]["legacy_model_id"] == "h2_indirect_period"
    assert ids["indirect_to_engagement"]["paper_label"] == "H2"
    assert ids["indirect_to_engagement"]["legacy_model_id"] == "h1_engagement_indirect"
    assert "not theoretically interesting" not in common.QUALITATIVE_EXCLUSION_REASONS


def test_a_eligibility_cols_exclude_engagement():
    forbidden = {
        "residual_engagement",
        "likes_resid",
        "comments_resid",
        "reposts_resid",
        "likes_count",
        "comments_count",
        "reposts_count",
        "raw_engagement",
        "log_engagement",
        "predicted_log_engagement",
    }
    assert not (q.A_ELIGIBILITY_COLS & forbidden)


def test_a_eligibility_ignores_residual_and_zero_engagement():
    df = pd.DataFrame(
        [
            _row(0, t2=0, indirect=1, narrative="Trauma-Help-Loss", residual=-9.0, likes=0),
            _row(1, t2=0, indirect=1, narrative="Trauma-Help-Loss", residual=9.0, likes=99),
            _row(
                2,
                t2=0,
                indirect=1,
                narrative="Trauma-Help-Loss",
                needs=True,
                residual=9.0,
            ),
        ]
    )
    mask = q.a_eligibility_mask(df)
    assert bool(mask.iloc[0]) and bool(mask.iloc[1])
    assert not bool(mask.iloc[2])
    df2 = df.copy()
    df2["residual_engagement"] = -df2["residual_engagement"]
    df2["likes_count"] = 0
    df2["raw_engagement"] = 0
    assert list(q.a_eligibility_mask(df2)) == list(mask)


def test_a1_matched_quotas_and_weather_both_years():
    df = _balanced_frame()
    a1 = q.draw_a1(df, seed=1)
    sel = a1.loc[a1["selected"]]
    for t2 in (0, 1):
        sub = sel.loc[pd.to_numeric(sel["t2"]) == t2]
        counts = sub["label_narrative_clean"].value_counts()
        assert int(counts.get("Trauma-Help-Loss", 0)) == 7
        assert int(counts.get("Memory-Reactivation", 0)) == 4
        assert int(counts.get("Ordinary-Weather-Info", 0)) == 4
        assert "Other-Unclear" not in set(sub["label_narrative_clean"])


def test_a2_six_two_and_indirect_subset_of_a1():
    df = _balanced_frame()
    a1 = q.draw_a1(df, seed=2)
    a2_ind = q.draw_a2_indirect_from_a1(a1, seed=2)
    a2_dir = q.draw_a2_direct(df, seed=2)
    sel_ind = a2_ind.loc[a2_ind["selected"]]
    sel_dir = a2_dir.loc[a2_dir["selected"]]
    a1_sel_mids = set(a1.loc[a1["selected"], "mid"].astype(str))
    assert set(sel_ind["mid"].astype(str)) <= a1_sel_mids
    for frame, ind_val in ((sel_ind, 1), (sel_dir, 0)):
        assert (pd.to_numeric(frame["indirect_clean"]) == ind_val).all()
        for t2 in (0, 1):
            sub = frame.loc[pd.to_numeric(frame["t2"]) == t2]
            counts = sub["label_narrative_clean"].value_counts()
            assert int(counts.get("Trauma-Help-Loss", 0)) == 6
            assert int(counts.get("Memory-Reactivation", 0)) == 2
            assert "Ordinary-Weather-Info" not in set(sub["label_narrative_clean"])


def test_profile_score_and_b_eligible_pool():
    df = pd.DataFrame(
        [
            _row(
                0,
                t2=0,
                indirect=1,
                narrative="Trauma-Help-Loss",
                likes_r=2,
                comments_r=2,
                reposts_r=-1,
            ),
            _row(
                1,
                t2=0,
                indirect=1,
                narrative="Trauma-Help-Loss",
                likes_r=0.001,
                comments_r=0.001,
                reposts_r=-0.001,
            ),
            _row(
                2,
                t2=0,
                indirect=1,
                narrative="Trauma-Help-Loss",
                likes_r=-1,
                comments_r=2,
                reposts_r=-1,
            ),
            _row(
                3,
                t2=1,
                indirect=1,
                narrative="Trauma-Help-Loss",
                likes_r=3,
                comments_r=1,
                reposts_r=0,
            ),
        ]
    )
    elig = q.bounded_resonance_eligible(df)
    assert list(elig) == [True, True, False, True]
    scored = df.loc[elig].copy()
    scored["score_b"] = q.profile_score_b(scored)
    t0 = scored.loc[scored["t2"] == 0].sort_values("score_b", ascending=False)
    assert str(t0.iloc[0]["mid"]) == "0"
    drawn = q.select_b(df, n_per_period=1, oversample=2, seed=1)
    sel_b = drawn.loc[drawn["selected"]]
    assert not sel_b.empty
    assert bool(q.bounded_resonance_eligible(sel_b).all())
    assert "score_b" in sel_b.columns


def test_select_c_period_p90_and_min_reposts():
    rng = np.random.default_rng(0)
    rows = []
    for t2 in (0, 1):
        for j in range(20):
            r = float(j)
            rows.append(
                _row(
                    t2 * 100 + j,
                    t2=t2,
                    indirect=1,
                    narrative="Trauma-Help-Loss",
                    reposts_r=r,
                    reposts=2 if j >= 18 else (1 if j >= 17 else 0),
                )
            )
    df = pd.DataFrame(rows)
    df["reposts_resid"] = df["reposts_resid"] + rng.normal(0, 0.01, len(df))
    c = q.select_c(df, n_per_period=2, oversample=2, min_reposts=2, seed=3)
    sel = c.loc[c["selected"]]
    assert (pd.to_numeric(sel["reposts_count"]) >= 2).all()
    assert set(pd.to_numeric(sel["t2"])) == {0, 1}
    for t2 in (0, 1):
        period = df.loc[pd.to_numeric(df["t2"]) == t2]
        cut = float(np.nanpercentile(period["reposts_resid"].to_numpy(dtype=float), 90))
        sub = sel.loc[pd.to_numeric(sel["t2"]) == t2]
        assert not sub.empty
        assert (pd.to_numeric(sub["reposts_resid"]) > cut).all()


def test_contrasts_period_median_and_p10():
    rows = []
    k = 0
    for t2 in (0, 1):
        for j in range(20):
            rows.append(
                _row(
                    k,
                    t2=t2,
                    indirect=1,
                    narrative="Trauma-Help-Loss",
                    pred=float(j),
                    residual=float(-j),
                )
            )
            k += 1
    df = pd.DataFrame(rows)
    neg = q.select_contrasts(df, n_per_period=2, oversample=2, seed=4)
    sel = neg.loc[neg["selected"]]
    for t2 in (0, 1):
        period = df.loc[df["t2"] == t2]
        med = float(period["predicted_log_engagement"].median())
        p10 = float(np.nanpercentile(period["residual_engagement"].to_numpy(), 10))
        sub = sel.loc[pd.to_numeric(sel["t2"]) == t2]
        assert (pd.to_numeric(sub["predicted_log_engagement"]) >= med).all()
        assert (pd.to_numeric(sub["residual_engagement"]) <= p10).all()


def test_unique_case_multirole_not_duplicated():
    a = pd.DataFrame(
        {
            "mid": ["1", "2"],
            "selection_role": [q.ROLE_A1, q.ROLE_A1],
            "selected": [True, True],
            "t2": [0, 1],
        }
    )
    b = pd.DataFrame(
        {
            "mid": ["1"],
            "selection_role": [q.ROLE_B],
            "selected": [True],
            "t2": [0],
        }
    )
    unique = q.merge_unique_cases([a, b])
    assert len(unique) == 2
    assert unique["mid"].is_unique
    row = unique.loc[unique["mid"] == "1"].iloc[0]
    roles = set(str(row["selection_role"]).split(";"))
    assert q.ROLE_A1 in roles and q.ROLE_B in roles


def test_exclusion_reason_closed_set():
    q.validate_exclusion_reason("")
    q.validate_exclusion_reason("coding_ambiguity")
    with pytest.raises(ValueError, match="disallowed"):
        q.validate_exclusion_reason("not theoretically interesting")


def test_selection_log_has_stratum():
    a1 = q.draw_a1(_balanced_frame(), seed=5)
    log = q.build_selection_log([a1])
    assert {"mid", "stratum", "eligible", "selected", "exclusion_reason"} <= set(log.columns)
    assert log["stratum"].astype(str).str.startswith("A1").all()
