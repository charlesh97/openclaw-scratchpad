# The Anatomy of a Decentralized Prediction Market — Microstructure Evidence from the Polymarket Order Book

**Source:** https://arxiv.org/abs/2604.24366  
**Published:** April 2026  
**Authors:** Multiple  
**Recommendation:** MEDIUM — Excellent empirical foundation, not directly implementable

## Summary

A comprehensive microstructure study of Polymarket using **30 billion WebSocket order-book events over 52 days**, joined with on-chain trade records across 600 markets. This is the most detailed Polymarket microstructure analysis ever published.

## Key Empirical Findings

1. **Longshot spread premium** — Longer odds have wider spreads (adverse selection by informed traders)
2. **Depth-concentration profile** — Order book depth follows a uniform geometric grid pattern (not top-of-book only)
3. **Null block-clock alignment** — No significant clustering around Ethereum block boundaries
4. **Broad maker diversity with concentrated tail** — Many liquidity providers but a few dominate
5. **Category-conditional spread differences** — Sports markets have wider spreads than crypto
6. **Median 50ms archive ingestion delay** — With multi-second tail (some events lag significantly)
7. **Self-counterparty wash share** — Median 1%, upper tail 22% (below typical exchange benchmarks)

## Why It Matters

- Provides the empirical baseline for evaluating our own market data
- The 50ms ingestion delay + multi-second tail is critical for latency-dependent strategies
- Wash trading estimates (1-22%) inform signal reliability
- Category-level spread differences help prioritize which markets to target

## Implementability: 2/5

- Pure empirical study — no code, no trading system
- But the stylized facts directly inform bot design:
  - Focus on crypto markets (narrower spreads, more volume)
  - Expect 50ms minimum ingestion latency
  - Plan for wash trading in volume figures
  - Geometric depth pattern enables better fill estimation

## Next Steps

1. Use depth-concentration findings to improve our order book fill probability model
2. Apply category-conditional spread analysis to market selection in our bot
3. Validate our latency measurements against their 50ms benchmark
4. Consider whether wash trading (1-22%) affects our PnL attribution
