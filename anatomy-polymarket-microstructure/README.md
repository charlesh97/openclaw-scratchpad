# The Anatomy of a Decentralized Prediction Market

**Source:** https://arxiv.org/abs/2604.24366  
**Authors:** Multiple (pre-print)  
**Published:** April 2026  
**Type:** Academic paper (empirical microstructure study)

## Summary

The most comprehensive microstructure study of Polymarket to date, using a **continuous tick-level archive of the public order-book feed** — 30 billion events over 52 days — joined to the on-chain trade record. Examines a pre-registered stratified panel of 600 markets.

### Eight Stylized Facts

1. **Longshot spread premium** — Longer odds have wider spreads
2. **Depth profile closer to uniform than top-of-book** — Liquidity is distributed deeper than typical equity markets
3. **Null block-clock alignment effect** — Block times don't predictably affect spreads
4. **Broad maker-wallet diversity with a concentrated tail** — Many unique makers, but a few dominate volume
5. **Category-conditional effective-spread differences** — Sports vs crypto vs politics have different microstructure
6. **Sub-50ms median archive ingestion delay** — But a multi-second tail (some data arrives late)
7. **Self-counterparty wash share: median 1%, 22% upper tail** — Some wash trading exists but not dominant
8. **Complete microstructure characterization** — Full order book dynamics mapped

## Why It Matters

- First empirical evidence that Polymarket's microstructure varies by market category
- Sub-50ms ingestion means fast bots can see the same data as the exchange
- Wash trading is non-trivial at the extreme (22% upper tail) but manageable at median
- Essential reading for building microstructure-aware trading strategies

## Implementability: 3/5

- Provides empirical parameters for strategy modeling
- Can inform optimal quoting behavior per market category
- Wash trading data helps calibrate risk models

## Next Steps
1. Use category-specific spread data to parameterize our MM strategies
2. Account for the multi-second ingestion tail in latency models
3. Monitor for wash trading patterns in our execution flow
