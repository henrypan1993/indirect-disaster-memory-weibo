"""Topic assignment on unique texts; merge topic_id to post-level (formal path).

By default writes analysis_ready_with_topics.csv (no entropy_norm).
Use --also-entropy to also write the legacy with_entropy table.
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
    PATH_ANALYSIS_ENTROPY,
    PATH_ANALYSIS_TOPICS,
    PATH_TOPIC_UNIQUE,
    PROJECT_ROOT,
    build_unique_corpus,
    compute_topic_entropy,
    merge_entropy_to_posts,
    merge_topics_to_posts,
    topics_frame_from_entropy_posts,
    utc_now_iso,
    write_json_report,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_ANALYSIS_BASE)
    p.add_argument("--output-unique", type=Path, default=PATH_TOPIC_UNIQUE)
    p.add_argument("--output-posts", type=Path, default=PATH_ANALYSIS_TOPICS)
    p.add_argument(
        "--also-entropy",
        action="store_true",
        help="Also write legacy analysis_ready_with_entropy.csv",
    )
    p.add_argument(
        "--entropy-output",
        type=Path,
        default=PATH_ANALYSIS_ENTROPY,
    )
    p.add_argument(
        "--from-entropy",
        type=Path,
        default=None,
        help="If set, strip entropy_norm from an existing with_entropy file "
        "instead of recomputing topics (freeze existing assignments).",
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

    if args.from_entropy is not None:
        src = args.from_entropy.expanduser().resolve()
        if not src.is_file():
            raise SystemExit(f"from-entropy not found: {src}")
        ent_df = pd.read_csv(src, dtype={"mid": str})
        topics_df = topics_frame_from_entropy_posts(ent_df)
        topics_df.to_csv(out_p, index=False, encoding="utf-8-sig")
        report = {
            "built_at": utc_now_iso(),
            "mode": "from_entropy_strip",
            "source_csv": str(src),
            "posts_csv": str(out_p),
            "n_rows": len(topics_df),
            "has_topic_id": "topic_id" in topics_df.columns,
            "has_entropy_norm": "entropy_norm" in topics_df.columns,
        }
        write_json_report(args.report, report)
        print(f"topics from frozen entropy table: {len(topics_df)} rows -> {out_p}")
        return

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str})
    uniq = build_unique_corpus(df)
    texts = uniq["text"].fillna("").astype(str).tolist()

    from common import ensure_embedding_cache

    embeddings = ensure_embedding_cache(df, model_name=args.model)
    topic_df, meta = compute_topic_entropy(texts, k=args.k, tau=args.tau, embeddings=embeddings)
    uniq_out = uniq[["text_hash"]].join(topic_df)
    uniq_out["k"] = args.k
    uniq_out["tau"] = args.tau

    out_u = args.output_unique.expanduser().resolve()
    out_u.parent.mkdir(parents=True, exist_ok=True)
    uniq_out.to_csv(out_u, index=False, encoding="utf-8-sig")

    merged_topics = merge_topics_to_posts(df, uniq_out)
    merged_topics.to_csv(out_p, index=False, encoding="utf-8-sig")

    entropy_path = None
    if args.also_entropy:
        merged_ent = merge_entropy_to_posts(df, uniq_out)
        entropy_path = args.entropy_output.expanduser().resolve()
        merged_ent.to_csv(entropy_path, index=False, encoding="utf-8-sig")

    report = {
        "built_at": utc_now_iso(),
        "mode": "compute",
        "input_csv": str(inp),
        "model": args.model,
        "unique_csv": str(out_u),
        "posts_csv": str(out_p),
        "entropy_csv": str(entropy_path) if entropy_path else None,
        **meta,
    }
    write_json_report(args.report, report)
    print(f"topic assignment: {len(uniq)} unique texts, K={args.k} -> {out_p}")


if __name__ == "__main__":
    main()
