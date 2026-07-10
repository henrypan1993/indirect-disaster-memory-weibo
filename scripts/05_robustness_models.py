"""Robustness checks aligned with main specs (all include T2 where applicable)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DEFAULT_K,
    DEFAULT_TAU,
    PATH_ANALYSIS_BASE,
    PATH_ANALYSIS_ENTROPY,
    PROJECT_ROOT,
    apply_sample_mask,
    build_unique_corpus,
    compute_topic_entropy,
    ensure_embedding_cache,
    load_embedding_cache_arrays,
    merge_entropy_to_posts,
    run_model_spec,
    spec_by_model_id,
    utc_now_iso,
)

OUT_MODELS = PROJECT_ROOT / "outputs" / "models"
ENTROPY_LONG = PROJECT_ROOT / "data" / "processed" / "entropy_robustness_long.csv"

ROBUSTNESS_MODEL_IDS = (
    "h1_engagement_indirect",
    "e1a_entropy_peripheral",
    "e1b_indirect_peripheral",
    "e2_entropy_increment",
)

H2A_ID = "h2a_entropy_reactivation"

# Full K x tau grid for entropy sensitivity (the (DEFAULT_K, DEFAULT_TAU) cell is
# the main spec and is therefore skipped to avoid duplicating the main results).
ENTROPY_K_GRID = (8, 10, 12)
ENTROPY_TAU_GRID = (0.05, 0.10, 0.20)

# Alternative peripheral percentile thresholds (main definition uses p90).
PERIPHERAL_PERCENTILES = (80, 90, 95)
PERIPHERAL_MODEL_IDS = (
    "e1a_entropy_peripheral",
    "e1b_indirect_peripheral",
    "h1_engagement_indirect",
)


def tau_tag(tau: float) -> str:
    return f"{int(round(tau * 100)):03d}"


def entropy_grid_specs() -> list[tuple[int, float, str]]:
    specs: list[tuple[int, float, str]] = []
    for k in ENTROPY_K_GRID:
        for tau in ENTROPY_TAU_GRID:
            if k == DEFAULT_K and abs(tau - DEFAULT_TAU) < 1e-9:
                continue
            specs.append((k, tau, f"k{k}_tau{tau_tag(tau)}"))
    return specs


def alt_peripheral_frame(df: pd.DataFrame, pct: int) -> tuple[pd.DataFrame, float, int]:
    """Return a copy of df with `peripheral` / `model_sample_h2` recomputed at pXX.

    Mirrors the contract rule `verified == 0 and followers < p{pct}_followers`,
    with the percentile computed once on the eligible corpus (include_main == 1).
    """
    out = df.copy()
    followers = pd.to_numeric(out["followers_count"], errors="coerce")
    eligible = out["include_main"] == 1
    threshold = float(np.nanpercentile(followers[eligible], pct))
    verified = pd.to_numeric(out["verified"], errors="coerce").fillna(0).astype(int)
    peripheral_alt = ((verified == 0) & (followers < threshold)).astype(int)
    out["peripheral"] = peripheral_alt
    out["model_sample_h2"] = eligible & (peripheral_alt == 1)
    return out, threshold, int(peripheral_alt[eligible].sum())


def dedupe_text_hash(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("mid").groupby("text_hash", as_index=False).first()


def attach_entropy_spec(
    base: pd.DataFrame,
    *,
    k: int,
    tau: float,
    embeddings,
) -> pd.DataFrame:
    uniq = build_unique_corpus(base)
    texts = uniq["text"].fillna("").astype(str).tolist()
    if embeddings is None:
        embeddings = load_embedding_cache_arrays(uniq)
    topic_df, _ = compute_topic_entropy(texts, k=k, tau=tau, embeddings=embeddings)
    uniq_ent = uniq[["text_hash"]].join(topic_df)
    return merge_entropy_to_posts(base, uniq_ent)


def run_specs_on_df(
    df: pd.DataFrame,
    model_ids: tuple[str, ...],
    spec_id: str,
    notes: str,
    *,
    y_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_overrides = y_overrides or {}
    for mid in model_ids:
        spec = spec_by_model_id(mid)
        y_col = y_overrides.get(mid)
        part, _, _ = run_model_spec(df, spec, spec_id, y_col=y_col, notes=notes)
        rows.extend(part)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_ANALYSIS_ENTROPY)
    p.add_argument("--base", type=Path, default=PATH_ANALYSIS_BASE)
    p.add_argument("--skip-entropy-rerun", action="store_true")
    p.add_argument(
        "--merge",
        action="store_true",
        help="Write all_models_summary.csv combining main + robustness.",
    )
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str})
    if "verified" in df.columns:
        df["verified"] = df["verified"].fillna(0).astype(int)

    rows: list[dict[str, Any]] = []
    entropy_long_parts: list[pd.DataFrame] = []

    rows.extend(
        run_specs_on_df(
            df.loc[df["high_clarity"]],
            ROBUSTNESS_MODEL_IDS,
            "high_clarity",
            "filter high_clarity==True",
        )
    )
    rows.extend(
        run_specs_on_df(
            df.loc[df["robust_no_review"]],
            ROBUSTNESS_MODEL_IDS,
            "robust_no_review",
            "filter robust_no_review==True",
        )
    )
    rows.extend(
        run_specs_on_df(
            dedupe_text_hash(df.loc[df["include_main"] == 1]),
            ("h1_engagement_indirect", "e1a_entropy_peripheral", "e1b_indirect_peripheral"),
            "unique_text",
            "one post per text_hash",
        )
    )

    for ycol, sid in (
        ("log_likes", "engagement_likes"),
        ("log_comments", "engagement_comments"),
        ("log_reposts", "engagement_reposts"),
    ):
        rows.extend(
            run_specs_on_df(
                df,
                ("h1_engagement_indirect",),
                sid,
                f"H1 y={ycol}",
                y_overrides={"h1_engagement_indirect": ycol},
            )
        )

    for pct in PERIPHERAL_PERCENTILES:
        alt_df, threshold, n_peripheral = alt_peripheral_frame(df, pct)
        sid = f"peripheral_p{pct}"
        print(
            f"peripheral threshold spec {sid} "
            f"(p{pct}_followers={threshold:.1f}, n_peripheral={n_peripheral}) ..."
        )
        rows.extend(
            run_specs_on_df(
                alt_df,
                PERIPHERAL_MODEL_IDS,
                sid,
                f"peripheral = verified==0 & followers<p{pct} ({threshold:.0f}); "
                f"n_peripheral={n_peripheral}",
            )
        )

    if not args.skip_entropy_rerun:
        base_path = args.base.expanduser().resolve()
        if base_path.is_file():
            base_df = pd.read_csv(base_path, dtype={"mid": str})
            specs = entropy_grid_specs()
            cached_emb = ensure_embedding_cache(base_df)
            for k, tau, sid in specs:
                print(f"robustness entropy spec {sid} (K={k}, tau={tau}) ...")
                ent_df = attach_entropy_spec(base_df, k=k, tau=tau, embeddings=cached_emb)
                entropy_long_parts.append(
                    ent_df[["mid", "entropy_norm"]].assign(spec_id=sid)
                )
                rows.extend(
                    run_specs_on_df(
                        ent_df.loc[ent_df["include_main"] == 1],
                        ("e1a_entropy_peripheral",),
                        sid,
                        f"E1a entropy K={k} tau={tau}",
                    )
                )
                rows.extend(
                    run_specs_on_df(
                        ent_df,
                        (H2A_ID,),
                        sid,
                        f"H2a entropy K={k} tau={tau}",
                    )
                )
                rows.extend(
                    run_specs_on_df(
                        apply_sample_mask(ent_df, "peripheral"),
                        ("e2_entropy_increment",),
                        sid,
                        f"E2 entropy K={k} tau={tau}",
                    )
                )

    if entropy_long_parts:
        long_df = pd.concat(entropy_long_parts, ignore_index=True)
        ENTROPY_LONG.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(ENTROPY_LONG, index=False, encoding="utf-8-sig")

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    built_at = utc_now_iso()
    robust_df = pd.DataFrame(rows)
    robust_df["built_at"] = built_at
    robust_path = OUT_MODELS / "robustness_models_summary.csv"
    robust_df.to_csv(robust_path, index=False, encoding="utf-8-sig")
    print(f"robustness -> {robust_path} ({len(rows)} rows)")

    if args.merge:
        main_path = OUT_MODELS / "main_models_summary.csv"
        appendix_path = OUT_MODELS / "appendix_without_t2.csv"
        parts = [robust_df]
        if main_path.is_file():
            parts.insert(0, pd.read_csv(main_path))
        if appendix_path.is_file():
            parts.append(pd.read_csv(appendix_path))
        all_df = pd.concat(parts, ignore_index=True)
        all_path = OUT_MODELS / "all_models_summary.csv"
        all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
        print(f"merged -> {all_path} ({len(all_df)} rows)")


if __name__ == "__main__":
    main()
