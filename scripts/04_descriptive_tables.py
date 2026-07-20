"""Descriptive tables and exploratory figures (not manuscript figure numbers)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PATH_ANALYSIS_TOPICS, PROJECT_ROOT

OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_ANALYSIS_TOPICS)
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str})
    main_df = df.loc[df["include_main"] == 1].copy()

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    # Table 1: sample structure
    t1 = (
        main_df.groupby(["wave", "account_group", "label_narrative_clean"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    t1.to_csv(OUT_TABLES / "table_1_sample_structure.csv", index=False, encoding="utf-8-sig")

    # Table 2: label distributions
    rows = []
    for col in ("label_narrative_clean", "label_emotion_clean", "label_expression_clean"):
        vc = main_df[col].value_counts(dropna=False)
        for val, cnt in vc.items():
            rows.append({"variable": col, "value": val, "n": int(cnt)})
    pd.DataFrame(rows).to_csv(
        OUT_TABLES / "table_2_label_distribution.csv", index=False, encoding="utf-8-sig"
    )

    # Table 3: core modeling variables
    num_cols = [
        "log_engagement",
        "log_followers",
        "text_length",
        "hashtag_count",
        "peripheral",
        "indirect_clean",
        "t2",
    ]
    num_cols = [c for c in num_cols if c in main_df.columns]
    desc = main_df[num_cols].describe().T.reset_index(names="variable")
    desc.to_csv(OUT_TABLES / "table_3_core_variables.csv", index=False, encoding="utf-8-sig")

    # Exploratory: narrative by wave x group
    ct = pd.crosstab(
        [main_df["wave"], main_df["account_group"]],
        main_df["label_narrative_clean"],
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, legend=True, fontsize=8)
    ax.set_title("Narrative by wave and account group")
    ax.set_xlabel("wave / account_group")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "exploratory_narrative_by_wave_group.png", dpi=150)
    plt.close(fig)

    # Exploratory: expression by group
    fig, ax = plt.subplots(figsize=(8, 5))
    main_df.groupby("account_group")["label_expression_clean"].value_counts(normalize=True).unstack(
        fill_value=0
    ).plot(kind="bar", ax=ax)
    ax.set_title("Expression distribution by account group")
    ax.set_ylabel("proportion")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "exploratory_expression_by_group.png", dpi=150)
    plt.close(fig)

    # Exploratory: engagement distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    main_df["log_engagement"].dropna().hist(bins=50, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("log_engagement distribution (include_main)")
    ax.set_xlabel("log_engagement")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "exploratory_log_engagement_distribution.png", dpi=150)
    plt.close(fig)

    print(f"descriptive outputs -> {OUT_TABLES} and {OUT_FIGURES}")


if __name__ == "__main__":
    main()
