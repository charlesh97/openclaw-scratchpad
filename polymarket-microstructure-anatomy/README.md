# Polymarket Order Book Microstructure (Anatomy Paper)

**Source:** https://arxiv.org/abs/2604.24366  
**Recommendation:** MEDIUM

## Summary

A comprehensive microstructure study of Polymarket using 30 billion order book events over 52 days, joined to on-chain trade records. Analyzes 600 markets across categories.

## Key Findings

1. **Longshot Spread Premium** — Longshot outcomes have wider spreads, consistent with traditional finance
2. **Depth Profile** — More uniform than top-of-book pattern, closer to geometric grid
3. **Null Block-Clock Effect** — Block production timing does not correlate with market activity
4. **Maker Diversity** — Broad wallet diversity with concentrated tail
5. **Category Spread Differences** — Effective spread varies significantly by market category
6. **Archive Delay** — Sub-50ms median ingestion delay with multi-second tail
7. **Wash Trading** — Median 1% self-counterparty trading (well below unregulated crypto benchmarks)
8. **Depth Determinants** — Market duration, price level, and volume explain depth; no residual time-to-close effect

## Critical Measurement Result

Trade direction inferred from Polymarket's public order-book feed agrees with on-chain ground truth on only ~59% of buckets — well below the ~80% accuracy of Lee-Ready on Nasdaq. Researchers must use on-chain OrderFilled events for trade direction.

## Why It Matters

Essential reading for anyone building trading bots on Polymarket. The measurement result about trade direction accuracy is critical — using the public feed alone will lead to incorrect conclusions. The paper provides a replication package with the join code.

## Implementability: 1/5 (Research paper, not implementation)

Pure research — no code to deploy. But the findings directly inform strategy design:

- Don't rely on order-book feed for trade direction
- Account for category-specific spread differences
- Design for the depth profile (uniform, not top-of-book)

## Next Steps

- Download replication package from GitHub/replication repo
- Validate findings on live data
- Use depth profile findings to design quoting strategy
