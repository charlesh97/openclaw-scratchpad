# The Anatomy of a Decentralized Prediction Market: Microstructure Evidence from the Polymarket Order Book

**Source:** arXiv:2604.24366 (April 2026) — https://arxiv.org/abs/2604.24366
**Recommendation:** YES — **TOP 2 for today's email**

## Summary

This landmark paper provides the most comprehensive microstructure analysis of Polymarket ever published. The authors study 30 billion tick-level WebSocket order-book events over 52 days, joined to on-chain trade records, across a pre-registered panel of 600 markets.

## Key Findings

### Eight Stylized Facts About Polymarket Microstructure:

1. **Longshot Spread Premium**: Bid-ask spreads are wider for extreme probabilities (near 0 or 1), consistent with traditional prediction market theory.

2. **Depth Distribution**: More uniform/geometric grid-shaped depth than conventional top-of-book pattern — not what most prediction market models assume.

3. **Null Block Clock Alignment**: No evidence of block-level timing patterns affecting order flow.

4. **Maker Diversity**: Broad wallet diversity with a concentrated tail (few wallets dominate).

5. **Category-Conditional Spreads**: Effective spreads vary significantly by market category (sports vs crypto vs politics).

6. **Ingestion Latency**: Median <50ms archive-ingestion delay, but with a multi-second tail — important for latency-sensitive strategies.

7. **Wash Trading**: Median 1% self-counterparty wash share, with 22% upper tail — well below unregulated crypto exchange benchmarks.

8. **Depth Decay Near Resolution**: Depth decays with log seconds-to-close (slope = 0.55, t=3.85).

### Critical Measurement Result

**Trade direction inferred from Polymarket's public order-book feed agrees with on-chain ground truth only ~59% of the time** — barely above 50% chance baseline. The effective half-spread changes sign between feed- and on-chain directions on 67% of markets.

**Bottom line**: Microstructure research MUST use on-chain OrderFilled events, not order-book feed inference. The authors release a replication package.

## Why It Matters

This is the definitive microstructure reference for Polymarket. It provides:
- Empirical benchmarks for strategy calibration (spreads, depth, latency)
- Essential correction about data quality (don't trust feed-inferred trade direction)
- Replication package for building your own microstructure analysis

## Implementability: 4/5

Not a trading strategy per se, but an essential reference. The stylized facts directly inform parameter choices for any market-making or arbitrage strategy on Polymarket.

## Next Steps

1. Download the replication package from GitHub (philippdubach/polymarket-microstructure)
2. Incorporate stylized facts into strategy parameter calibration
3. Use the on-chain trade direction joining methodology in our own backtests
4. Benchmark our strategies against the reported liquidity and spread profiles
