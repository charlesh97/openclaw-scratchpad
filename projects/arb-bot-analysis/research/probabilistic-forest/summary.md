# Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets

**Source:** https://arxiv.org/abs/2508.03474 (Aug 2025)

**Authors:** Oriol Saguillo et al.

## Summary

An empirical arbitrage analysis on Polymarket covering 17,218 conditions. The researchers found **7,051 conditions with single-market arbitrage** and an estimated **$40M USD of profit extracted** by arbitrageurs.

### Two Distinct Forms of Arbitrage Identified
1. **Market Rebalancing Arbitrage** — Occurs within a single market/condition when YES+NO != $1
2. **Combinatorial Arbitrage** — Spans across multiple related markets/conditions

### Methodology
- Used heuristic-driven reduction strategy (O(2^n) naive → tractable via timeliness, topical similarity, combinatorial relationships)
- Expert validation of matches
- On-chain order book data analysis

### Key Stats
- $40M realized profit extracted
- 7,051 conditions had single-market arbitrage
- 17,218 total conditions analyzed

## Implementability: 5/5

The combinatorial arbitrage detection algorithm is directly applicable. The heuristic reduction strategy solves the combinatorial explosion problem.

## Recommendation: MEDIUM

Must-read for understanding the arbitrage landscape. The combinatorial detection algorithm should inform our bot's core detection engine.
