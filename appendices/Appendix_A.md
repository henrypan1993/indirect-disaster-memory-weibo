# Appendix A. Data Retrieval and Corpus Construction

## A.1 Time windows and keywords

**Table A1. Retrieval windows**

| Wave | Role | Date range | Window IDs |
|------|------|------------|------------|
| T1 | 2021 disaster period | 18 July – 7 August 2021 | `t1_w1`–`t1_w5` |
| T2 | 2025 reactivation period | 4–17 August 2025 | `t2_main_w1`–`t2_main_w3` |

T2 is defined by renewed heavy-rainfall / flooding / public-safety risk in summer 2025, not by calendar anniversary alone. T1 extends beyond peak disaster days to include short-run aftermath discourse.

**Keywords (OR-combined, both waves):** 郑州暴雨 (Zhengzhou rainstorm); 郑州洪水 (Zhengzhou flood); 郑州7·20 (Zhengzhou 7/20); 郑州内涝 (Zhengzhou urban flooding). Auxiliary commemorative query tracks are excluded from the main corpus.

## A.2 Retrieval, merge, and screening

Historical search produced mid-level identifiers. Unique mids were backfilled via `statuses/show_batch`, merged into an analysis-ready wide table, screened for accessibility, and filtered through Gate then Core annotation.

**Table A2. Sample attrition**

| Step | Records | Exclusion / criterion |
|------|--------:|------------------------|
| Initial retrieved mids | 62,513 | Keyword and date-window retrieval; deduplicated by `mid` |
| Backfilled records | 38,980 | Status objects returned by `statuses/show_batch` |
| Merged records | 39,654 | Wide-table merge and field alignment |
| Accessible records | 24,796 | Visible survivor subset (Gate input) |
| Core-labeled corpus | 17,143 | `label_relevance = 1` and `score_relevance = 2` |

The increase from 38,980 to 39,654 reflects engineering merge and field alignment, not a new retrieval wave. Gate filtering removed 7,653 of 24,796 visible posts.

## A.3 Survivor-corpus boundary

The main sample comprises posts that remained accessible at collection, were successfully backfilled, and passed Gate/Core screening. Deleted, inaccessible, or never-retrieved posts are outside the sample. Engagement outcomes measure interaction **within this survivor corpus** and do not measure algorithmic exposure, recommendation intensity, or platform-wide survival probability.

## A.4 Deduplication and cleaning

| Layer | Rule |
|-------|------|
| API / log | Deduplicate by `mid` |
| Wide table | Deduplicate by `dedup_key` / `mid` |
| Visibility | Retain visible posts with non-empty analysis text |
| Labels | Harmonize within `text_hash × wave` by strict majority vote |

`text_hash` is SHA-256 of UTF-8 stripped analysis text. Cleaned Core table: N = 17,143; high-clarity = 16,977; needs manual review = 307.
