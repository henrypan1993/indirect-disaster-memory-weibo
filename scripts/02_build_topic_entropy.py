"""Topic space + normalized entropy on unique texts; merge to post-level."""

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
    PATH_TOPIC_UNIQUE,
    PROJECT_ROOT,
    build_unique_corpus,
    compute_topic_entropy,
    merge_entropy_to_posts,
    utc_now_iso,
    write_json_report,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=PATH_ANALYSIS_BASE)
    p.add_argument("--output-unique", type=Path, default=PATH_TOPIC_UNIQUE)
    p.add_argument("--output-posts", type=Path, default=PATH_ANALYSIS_ENTROPY)
    p.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "topic_entropy_report.json",
    )
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--model", type=str, default=DEFAULT_EMBED_MODEL)
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    df = pd.read_csv(inp, dtype={"mid": str})
    uniq = build_unique_corpus(df)
    texts = uniq["text"].fillna("").astype(str).tolist()

    from common import ensure_embedding_cache

    embeddings = ensure_embedding_cache(df, model_name=args.model)
    topic_df, meta = compute_topic_entropy(
        texts, k=args.k, tau=args.tau, embeddings=embeddings
    )
    uniq_out = uniq[["text_hash"]].join(topic_df)
    uniq_out["k"] = args.k
    uniq_out["tau"] = args.tau

    out_u = args.output_unique.expanduser().resolve()
    out_u.parent.mkdir(parents=True, exist_ok=True)
    uniq_out.to_csv(out_u, index=False, encoding="utf-8-sig")

    merged = merge_entropy_to_posts(df, uniq_out)
    out_p = args.output_posts.expanduser().resolve()
    merged.to_csv(out_p, index=False, encoding="utf-8-sig")

    report = {
        "built_at": utc_now_iso(),
        "input_csv": str(inp),
        "model": args.model,
        "unique_csv": str(out_u),
        "posts_csv": str(out_p),
        **meta,
    }
    write_json_report(args.report, report)
    print(f"topic entropy: {len(uniq)} unique texts, K={args.k} -> {out_p}")


if __name__ == "__main__":
    main()
