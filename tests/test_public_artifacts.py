"""Checks that curated public tables match the revised manuscript aggregates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "outputs" / "public"
FORBIDDEN = {
    "mid",
    "account_id",
    "text",
    "text_for_coding",
    "analysis_text",
    "analysis_text_final",
    "username",
    "screen_name",
    "label_expression",
    "llm_label_expression",
}


def test_public_tree_has_no_row_level_validation_or_identifiers() -> None:
    assert not (ROOT / "data" / "public" / "expression_validation_pairs.csv").exists()
    for folder in (PUBLIC, ROOT / "data" / "public"):
        for path in folder.glob("*.csv"):
            columns = {
                name.strip() for name in path.read_text(encoding="utf-8").splitlines()[0].split(",")
            }
            assert columns.isdisjoint(FORBIDDEN), path.name


def test_q1_adjusted_or_and_q2_broader_coefficient() -> None:
    period = pd.read_csv(PUBLIC / "table_h2_m1_m2_comparison.csv")
    adjusted = period.loc[period["spec_id"] == "h2_m2_composition"].iloc[0]
    assert round(float(adjusted["or"]), 2) == 1.46
    assert int(adjusted["n"]) == 5804

    broader = pd.read_csv(PUBLIC / "table_h1_broader_sample.csv").iloc[0]
    assert round(float(broader["coef"]), 3) == 0.132
    assert int(broader["n"]) == 10806

    components = pd.read_csv(PUBLIC / "table_q2_engagement_components_broader.csv")
    by_spec = components.set_index("spec_id")
    assert round(float(by_spec.loc["engagement_likes_broader", "coef"]), 3) == 0.102
    assert round(float(by_spec.loc["engagement_comments_broader", "coef"]), 3) == 0.070
    assert round(float(by_spec.loc["engagement_reposts_broader", "coef"]), 3) == 0.007
    assert int(by_spec.loc["engagement_likes_broader", "n"]) == 10806


def test_stacked_wald_matches_manuscript_core_sample() -> None:
    wald = pd.read_csv(PUBLIC / "table_engagement_asymmetry_wald.csv").set_index("contrast")
    assert round(float(wald.loc["joint_equality", "stat"]), 2) == 7.15
    assert int(wald.loc["joint_equality", "df_num"]) == 2
    assert int(wald.loc["joint_equality", "df_denom"]) == 5475
    assert round(float(wald.loc["likes_vs_reposts", "stat"]), 2) == 14.07
    assert round(float(wald.loc["comments_vs_reposts", "stat"]), 2) == 6.23
    assert round(float(wald.loc["likes_vs_comments", "stat"]), 2) == 3.46


def test_published_misclassification_aggregates_match_appendix_b() -> None:
    shares = pd.read_csv(PUBLIC / "table_misclassification_sensitivity_shares.csv")
    wave = shares.loc[shares["scenario"] == "validation_differential"].iloc[0]
    assert round(100 * float(wave["share_2021"]), 1) == 21.5
    assert round(100 * float(wave["share_2025"]), 1) == 54.2
    odds = pd.read_csv(PUBLIC / "table_misclassification_sensitivity_or.csv")
    fixed = odds.loc[
        (odds["scenario"] == "validation_differential")
        & (odds["uncertainty"] == "imputation_at_fixed_rates")
    ].iloc[0]
    assert int(fixed["n_reps"]) == 80
    assert round(float(fixed["median_or"]), 2) == 3.39
    counts = pd.read_csv(ROOT / "data" / "public" / "h2_indirect_counts.csv").set_index("period")
    assert int(counts.loc["T1", "n_indirect"]) == 675
    assert int(counts.loc["T1", "n"]) == 4827
    assert int(counts.loc["T2", "n_indirect"]) == 300
    assert int(counts.loc["T2", "n"]) == 1019
