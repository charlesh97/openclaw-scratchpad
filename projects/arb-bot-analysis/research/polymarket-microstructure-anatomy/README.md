# The Anatomy of a Decentralized Prediction Market

**Source:** https://arxiv.org/abs/2604.24366  
**Recommendation:** MEDIUM  
**Published:** April 27, 2026  

## Summary

The first comprehensive tick-level microstructure study of Polymarket using 30 billion order-book events over 52 days across 600 pre-registered markets. Provides the foundational empirical evidence for understanding Polymarket market structure, latency dynamics, and trader behavior.

## Key Findings (8 Stylized Facts)

1. **Longshot Spread Premium** — longshot outcomes have wider bid-ask spreads
2. **Uniform Depth Profile** — order book depth is closer to uniform than concentrated at top-of-book
3. **No Block-Clock Alignment** — no evidence of block timing manipulation
4. **Broad Maker Diversity** — many unique wallets providing liquidity, but with a concentrated tail
5. **Category-Conditional Spreads** — effective spreads vary significantly by market category
6. **Sub-50ms Median Ingestion** — median archive delay is <50ms but with multi-second tail
7. **1% Median Wash Share** — self-counterparty trades at median 1%, upper tail at 22%
8. **Wash Trading Below CEX Benchmarks** — well below traditional exchange wash rates (Cong et al.)

## Why It Matters

- First large-scale microstructure paper with actual order-book data from Polymarket
- 30 billion events analyzed — highest granularity study to date
- Directly relevant for building latency-sensitive strategies
- Wash trading data is important for strategy replication reliability

## Implications for Trading

- Median 50ms ingestion delay means fast arbitrage is possible but tail latency hurts
- 1% wash share is noise-level; 22% tail means some markets have significant fake volume
- Category spreads vary — some categories are more profitable for market-making
- Uniform depth profile means you can trade beyond top-of-book with less impact than expected

## Implementability: 3/5

Academic paper. Provides essential context but no executable code. Must build own implementation.

## Status: REFERENCE
