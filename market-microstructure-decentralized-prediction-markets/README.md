# market-microstructure-decentralized-prediction-markets (SoK)

**Source:** https://arxiv.org/abs/2510.15612
**Published:** October 2025 (updated March 2026)
**Recommendation:** MEDIUM (strong survey, informs architecture)

## What it does

A **Systematization of Knowledge (SoK)** paper — a comprehensive survey of market microstructure research in Decentralized Prediction Markets (DePMs) including Polymarket. Covers:

- Order book dynamics in AMM-based prediction markets
- How liquidity provision works differently than in traditional limit order book markets
- Taxonomy of arbitrage mechanisms (market rebalancing vs. combinatorial)
- Empirical findings on when and how arbitrage breaks down
- Regulatory considerations across jurisdictions

## Why it matters

- Provides the theoretical framework needed to understand *why* arbitrage opportunities exist and persist in prediction markets
- Directly relevant to building a sustainable arbitrage system — explains the structural conditions that give rise to mispricings
- Cites Tsang & Yang (2026) on microstructure analysis of the 2024 US presidential election — grounding in real Polymarket data

## Key Reference

The Tsang and Yang (2026) single-event time-series microstructure study is cited as the prior empirical work. Our NBA arbitrage paper (2605.00864) appears to extend this methodology.

## Paper URL
https://arxiv.org/abs/2510.15612

## Implementability: N/A (survey paper)

Not a bot, but essential reading for anyone building a serious prediction market arbitrage system.

## Next Steps

1. Download and read the full SoK
2. Use the taxonomy of DePM microstructure to identify which market types have the most persistent arbitrage opportunities
3. Incorporate the regulatory analysis into compliance planning for cross-platform arbitrage (Kalshi CFTC considerations)

---
*Part of vega's arb-bot-analysis research.*