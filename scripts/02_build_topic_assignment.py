"""Topic assignment on unique texts; merge topic_id to posts (formal path).

Writes analysis_ready_with_topics.csv with topic_id only (no entropy_norm).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_K,
    DEFAULT_TAU,
    PATH_ANALYSIS_BASE,
    PATH_ANALYSIS_TOPICS,
    PATH_TOPIC_UNIQUE,
    PROJECT_ROOT,
    assign_topics,
    build_unique_corpus,
    merge_topics_to_posts,
    topics_frame_from_legacy_posts,
    utc_now_iso,
    write_json_report,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_ANALYSIS_BASE)
    p.add_argument("--output-unique", type=Path, default=PATH_TOPIC_UNIQUE)
    p.add_argument("--output-posts", type=Path, default=PATH_ANALYSIS_TOPICS)
    p.add_argument(
        "--from-legacy-posts",
        type=Path,
        default=None,
        help="If set, strip optional entropy_norm from a legacy posts table "
        "instead of recomputing topics (freeze existing topic_id).",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "topic_assignment_report.json",
    )
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--model", type=str, default=DEFAULT_EMBED_MODEL)
    args = p.parse_args()

    out_p = args.output_posts.expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if args.from_legacy_posts is not None:
        src = args.from_legacy_posts.expanduser().resolve()
        if not src.is_file():
            raise SystemExit(f"from-legacy-posts not found: {src}")
        legacy_df = pd.read_csv(src, dtype={"mid": str})
        topics_df = topics_frame_from_legacy_posts(legacy_df)
        topics_df.to_csv(out_p, index=False, encoding="utf-8-sig")
        report = {
            "built_at": utc_now_iso(),
            "mode": "from_legacy_strip",
            "source_csv": str(src),
            "posts_csv": str(out_p),
            "n_rows": len(topics_df),
            "has_topic_id": "topic_id" in topics_df.columns,
            "has_entropy_norm": "entropy_norm" in topics_df.columns,
        }
        write_json_report(args.report, report)
        print(f"topics from legacy posts table: {len(topics_df)} rows -> {out_p}")
        return

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str})
    uniq = build_unique_corpus(df)
    texts = uniq["text"].fillna("").astype(str).tolist()

    from common import ensure_embedding_cache

    embeddings = ensure_embedding_cache(df, model_name=args.model)
    topic_df, meta = assign_topics(texts, k=args.k, tau=args.tau, embeddings=embeddings)
    uniq_out = uniq[["text_hash"]].join(topic_df)
    uniq_out["k"] = args.k
    uniq_out["tau"] = args.tau

    out_u = args.output_unique.expanduser().resolve()
    out_u.parent.mkdir(parents=True, exist_ok=True)
    uniq_out.to_csv(out_u, index=False, encoding="utf-8-sig")

    merged_topics = merge_topics_to_posts(df, uniq_out)
    merged_topics.to_csv(out_p, index=False, encoding="utf-8-sig")

    report = {
        "built_at": utc_now_iso(),
        "mode": "compute",
        "input_csv": str(inp),
        "model": args.model,
        "unique_csv": str(out_u),
        "posts_csv": str(out_p),
        **meta,
    }
    write_json_report(args.report, report)
    print(f"topic assignment: {len(uniq)} unique texts, K={args.k} -> {out_p}")


if __name__ == "__main__":
    main()
