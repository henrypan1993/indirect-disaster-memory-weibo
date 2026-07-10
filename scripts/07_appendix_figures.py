"""Appendix figures.

Figure A1: Sample construction flow (fixed pipeline counts from Appendix A.4).
Figure A2: T1 vs T2 expression and entropy comparison on the trauma x peripheral
           sample (same scope as H2a / H2b); descriptive only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PATH_ANALYSIS_ENTROPY, PROJECT_ROOT, apply_sample_mask

OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLOR_T1 = "#6b9ac4"
COLOR_T2 = "#d98a5f"

# Fixed pipeline counts and stage descriptions (Appendix A.4).
FLOW_STAGES: tuple[tuple[str, str], ...] = (
    ("62,513", "Historical retrieval\n(deduplicated by mid)"),
    ("38,980", "show_batch status backfill\n(available records)"),
    ("39,654", "Analysis-ready wide table\n(merge & field alignment)"),
    ("24,796", "Visible subset\n(Gate LLM input)"),
    ("17,143", "Gate pass\n(relevant & relevance score = 2)"),
    ("17,143", "Core annotation & cleaning\n(main analysis sample)"),
)

SURVIVOR_NOTE = (
    "Survivor-corpus boundary: the analysis sample contains only posts that were still "
    "accessible, retrievable via show_batch, and passed the Gate/Core pipeline at "
    "collection time. Engagement reflects interaction within surviving posts, not "
    "exposure, recommendation, or platform-retention probability."
)


def wilson_ci_errors(
    props: list[float], counts: list[int], *, z: float = 1.96
) -> tuple[list[float], list[float]]:
    """Return (lower_err, upper_err) for 95% Wilson score intervals on proportions."""
    lower_err: list[float] = []
    upper_err: list[float] = []
    for p, n in zip(props, counts, strict=True):
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        lower_err.append(p - (center - half))
        upper_err.append((center + half) - p)
    return lower_err, upper_err


def fig_a1_sample_flow(out_path: Path) -> None:
    n = len(FLOW_STAGES)
    fig, ax = plt.subplots(figsize=(8.6, 10.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, n * 2 + 1.6)
    ax.axis("off")

    box_w, box_h = 6.4, 1.25
    x_center = 4.6
    y_positions = [(n - i) * 2 for i in range(n)]

    for i, ((count, label), y) in enumerate(zip(FLOW_STAGES, y_positions, strict=True)):
        box = FancyBboxPatch(
            (x_center - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.2,
            edgecolor="#33485c",
            facecolor="#eef3f8",
        )
        ax.add_patch(box)
        ax.text(
            x_center,
            y + 0.20,
            count,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="#16243a",
        )
        ax.text(
            x_center,
            y - 0.34,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#333333",
        )
        if i < n - 1:
            y_next = y_positions[i + 1]
            ax.add_patch(
                FancyArrowPatch(
                    (x_center, y - box_h / 2),
                    (x_center, y_next + box_h / 2),
                    arrowstyle="-|>",
                    mutation_scale=15,
                    linewidth=1.2,
                    color="#555555",
                )
            )

    # Explain the non-monotonic step (38,980 -> 39,654).
    y_mid = (y_positions[1] + y_positions[2]) / 2
    ax.annotate(
        "Non-monotonic step (+674):\nengineering merge & record\nalignment, not new collection",
        xy=(x_center + box_w / 2, y_mid),
        xytext=(x_center + box_w / 2 + 0.5, y_mid),
        ha="left",
        va="center",
        fontsize=8.0,
        style="italic",
        color="#b5532a",
        arrowprops={"arrowstyle": "-", "color": "#b5532a", "linewidth": 0.9},
    )

    fig.text(
        0.5,
        0.045,
        SURVIVOR_NOTE,
        ha="center",
        va="top",
        fontsize=7.6,
        style="italic",
        color="#444444",
        wrap=True,
    )
    fig.subplots_adjust(bottom=0.13, top=0.97)
    fig.savefig(out_path)
    plt.close(fig)


def fig_a2_t1_t2(df: pd.DataFrame, out_path: Path) -> dict[str, int]:
    sub = apply_sample_mask(df, "trauma_peripheral").copy()
    sub["t2"] = pd.to_numeric(sub["t2"], errors="coerce")

    expr = sub.dropna(subset=["indirect_clean", "t2"])
    grp = expr.groupby("t2")["indirect_clean"].agg(["mean", "count"])
    props = [float(grp.loc[0, "mean"]), float(grp.loc[1, "mean"])]
    n_expr = [int(grp.loc[0, "count"]), int(grp.loc[1, "count"])]
    lower_err, upper_err = wilson_ci_errors(props, n_expr)

    ent = sub.dropna(subset=["entropy_norm", "t2"])
    ent_t1 = ent.loc[ent["t2"] == 0, "entropy_norm"].to_numpy()
    ent_t2 = ent.loc[ent["t2"] == 1, "entropy_norm"].to_numpy()
    n_ent = [int(ent_t1.size), int(ent_t2.size)]

    labels = ["T1 (2021 disaster)", "T2 (2025 reactivation)"]
    colors = [COLOR_T1, COLOR_T2]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.4, 4.9))

    # Panel A: indirect / mixed expression proportion (95% Wilson CI error bars).
    bars = ax_a.bar(
        labels,
        props,
        yerr=[lower_err, upper_err],
        capsize=5,
        color=colors,
        edgecolor="#333333",
    )
    top_ci = max(p + ue for p, ue in zip(props, upper_err, strict=True))
    ax_a.set_ylim(0, top_ci * 1.22)
    ax_a.set_ylabel("Share of indirect/mixed expression")
    ax_a.set_title("Panel A. Indirect/mixed expression", fontsize=11)
    for bar, p, ue in zip(bars, props, upper_err, strict=True):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            p + ue + top_ci * 0.04,
            f"{p:.1%}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax_a.set_xticks(range(len(labels)))
    ax_a.set_xticklabels([f"{lab}\nn = {nn:,}" for lab, nn in zip(labels, n_expr, strict=True)])
    ax_a.grid(axis="y", linestyle=":", alpha=0.4)

    # Panel B: normalized entropy distribution.
    bp = ax_b.boxplot(
        [ent_t1, ent_t2],
        patch_artist=True,
        showmeans=True,
        widths=0.55,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#333333"},
        medianprops={"color": "#222222", "linewidth": 1.4},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.25},
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#333333")
    ax_b.set_ylabel("Normalized topic entropy")
    ax_b.set_title("Panel B. Normalized topic entropy", fontsize=11)
    ax_b.set_xticks([1, 2])
    ax_b.set_xticklabels([f"{lab}\nn = {nn:,}" for lab, nn in zip(labels, n_ent, strict=True)])
    ax_b.grid(axis="y", linestyle=":", alpha=0.4)

    fig.subplots_adjust(bottom=0.12, top=0.92, wspace=0.28)
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "n_expr_t1": n_expr[0],
        "n_expr_t2": n_expr[1],
        "n_ent_t1": n_ent[0],
        "n_ent_t2": n_ent[1],
        "prop_indirect_t1": round(props[0], 4),
        "prop_indirect_t2": round(props[1], 4),
        "entropy_mean_t1": round(float(ent_t1.mean()), 4),
        "entropy_mean_t2": round(float(ent_t2.mean()), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_ANALYSIS_ENTROPY)
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str})
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    a1_path = OUT_FIGURES / "fig_A1_sample_construction_flow.png"
    a2_path = OUT_FIGURES / "fig_A2_t1_t2_expression_entropy.png"

    fig_a1_sample_flow(a1_path)
    stats = fig_a2_t1_t2(df, a2_path)

    print(f"figure A1 -> {a1_path}")
    print(f"figure A2 -> {a2_path}")
    print("A2 descriptive stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
