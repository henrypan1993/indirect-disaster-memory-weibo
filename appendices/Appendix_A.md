# Appendix A. Corpus Construction and Attrition

This appendix documents corpus construction and sample attrition.

## A.1 Retrieval windows

**Table A1. Retrieval windows**

| Wave | Role | Date range |
|------|------|------------|
| T1 | 2021 disaster period | 18 July – 7 August 2021 |
| T2 | 2025 heavy-rainfall context | 4–17 August 2025 |

T2 is defined by renewed heavy-rainfall / flooding / public-safety risk in summer 2025, not by calendar anniversary alone. T1 extends beyond peak disaster days to include short-run aftermath discourse.

**Keywords (OR-combined, both waves):** 郑州暴雨 (Zhengzhou rainstorm); 郑州洪水 (Zhengzhou flood); 郑州7·20 (Zhengzhou 7/20); 郑州内涝 (Zhengzhou urban flooding). Auxiliary commemorative query tracks are excluded from the main corpus.

## A.2 Retrieval and screening

Posts were retrieved through keyword and date-window searches, followed by batch backfilling, record consolidation, visibility screening, and two-stage LLM annotation. Across seven retrieval windows, 63,858 retrieval entries yielded 62,513 unique post identifiers after deduplication.

**Table A2. Corpus construction and sample attrition**

| Step | Records | Criterion or processing outcome |
|------|--------:|---------------------------------|
| Raw retrieval entries | 63,858 | Entries collected across seven keyword-and-date retrieval windows |
| Unique retrieved identifiers | 62,513 | Unique post IDs after deduplication across retrieval windows |
| April 11 backfill records | 38,980 | Status records returned from 882 API calls on April 11, 2026 |
| Additional retained records | 674 | Non-overlapping status records from 39 batch files retained from April 10 |
| Consolidated posts | 39,654 | Unique posts after combining both sets of backfill records |
| Visible posts | 24,801 | Posts with state_code = 0 |
| Gate input corpus | 24,796 | Visible posts with nonempty analytical text and no pending review |
| Final Core corpus | 17,143 | Gate-relevant posts with llm_label_relevance = 1 and llm_score_relevance = 2 |

A backfill conducted on April 11, 2026, returned 38,980 status records. During consolidation, 674 additional non-overlapping records from 39 batch files retained from the previous day's backfill were included, yielding 39,654 unique posts. Deduplication removed none of those 674 posts. The completing run restarted batch-file numbering, so the earlier files were not overwritten.

Of the 24,801 visible posts, five were excluded because their final analytical text was empty or unusable, including one placeholder repost. The remaining 24,796 posts constituted the Gate input corpus. Gate screening excluded 7,653 posts: 2,620 were classified as irrelevant (relevance label 0, score 0), and 5,033 received relevance label 1 with score 1. All 17,143 posts that kept relevance label 1 and score 2 entered Core annotation. Labels were harmonized across duplicate texts within each wave by majority vote.

The increase from 38,980 to 39,654 reflects the inclusion of 674 distinct status records retained from the previous day's backfill. It does not result from duplicate records, additional retrieval during table merging, or field alignment.

## A.3 Corpus boundary

The final annotated corpus comprised 17,143 posts that were retrievable through the selected keywords and date windows, remained accessible at collection, contained usable analytical text, and met the Gate relevance criteria. Deleted, inaccessible, and never-retrieved posts were excluded, as were accessible posts that failed the text or Gate screens.

Engagement outcomes reflect observed platform interactions with posts in this survivor corpus. They do not directly measure algorithmic exposure, recommendation intensity, audience composition, moderation history, or platform-wide content survival. Inferences are restricted to the observable annotated corpus.
