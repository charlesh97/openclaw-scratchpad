# The Anatomy of a Decentralized Prediction Market: Microstructure Evidence from the Polymarket Order Book

**Source:** https://arxiv.org/abs/2604.24366
**Published:** April 2026

## Summary

A landmark microstructure study of Polymarket using **30 billion order book events over 52 days** joined to on-chain trade records. First paper to combine tick-level order book data with authoritative on-chain data for a prediction market.

### 8 Stylized Facts

1. **Longshot spread premium**: Wider spreads for extreme probabilities
2. **Uniform depth profile**: Closer to a geometric grid than top-of-book concentration
3. **Null block-clock alignment**: No evidence of time-based manipulation
4. **Broad maker diversity**: Many unique wallets, but concentrated in a few large ones
5. **Category-conditional effective spreads**: Sports vs. crypto vs. political markets differ significantly
6. **Sub-50ms median ingestion delay**: But with multi-second tail — some data arrives late
7. **Self-counterparty wash share**: Median 1%, upper tail 22% — well below crypto norms
8. **Depth explained by duration + price + volume**: No residual time-to-close effect

### Critical Measurement Finding
Trade direction inferred from Polymarket's public order book agrees with on-chain ground truth only **~59% of the time** (vs. ~80% for Nasdaq via Lee-Ready). This means microstructure research **must** use on-chain OrderFilled events, not the public feed.

### Replication Package
Code released at https://github.com/philippdubach/polymarket-microstructure

## Implications for Trading
- Order book data alone is insufficient for accurate trade direction analysis
- Liquidity is more evenly distributed across price levels than in traditional markets
- Wash trading exists but at lower levels than unregulated crypto venues
- Category-specific strategies are warranted (sports ≠ politics ≠ crypto)

## Implementability: 2/5
Academic paper with complex statistical methodology. The replication code is in Python but requires joining massive datasets.

## Next Steps
1. Download the replication package and run on a recent 7-day window
2. Compare findings with current market structure (post-Feb 2026 changes)
3. Use the trade direction methodology for our own arbitrage analysis
