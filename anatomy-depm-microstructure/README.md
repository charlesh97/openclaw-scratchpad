# The Anatomy of a Decentralized Prediction Market

**Source:** arXiv:2604.24366 — https://arxiv.org/abs/2604.24366
**Authors:** (Multi-author)
**Published:** April 2026

## Key Findings
- First tick-level microstructure study of Polymarket using **30 billion WebSocket events over 52 days**
- **8 stylized facts** about Polymarket:
  1. Longshot spread premium exists
  2. Depth-concentration closer to uniform geometric grid than top-of-book
  3. Null block-clock alignment effect (market timing doesn't impact microstructure)
  4. Broad maker-wallet diversity with concentrated tail
  5. Category-conditional differences in effective spread
  6. Sub-50ms median archive ingestion delay (but multi-second tail)
  7. Self-counterparty wash share: median 1%, 22% upper tail
  8. Cross-referenced to Cong et al. benchmarks

## Why It Matters
This is the most comprehensive microstructure analysis of Polymarket ever published. The sub-50ms ingestion delay and depth-concentration findings directly inform bot strategy design (optimal quoting levels, latency budgets).

## Implementability: 4/5
The stylized facts directly inform market making strategy. The depth-concentration grid finding suggests optimal quote placement strategies. The wash trading analysis validates cleanliness of the Polymarket order book.

## Next Steps
1. Replicate the depth-concentration analysis on current Polymarket data
2. Design quoting strategy using the geometric grid finding
3. Monitor wash trading patterns for market quality signals
