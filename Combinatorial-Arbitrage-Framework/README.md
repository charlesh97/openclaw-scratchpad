# Combinatorial Arbitrage Framework

**Source:** [arXiv:2508.03474 - Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets](https://arxiv.org/abs/2508.03474)
**Authors:** Oriol Saguillo, V. Ghafouri, L. Kiffer, G. Suarez-Tangil (IMDEA Networks Institute)
**Date:** August 2025
**Recommendation:** ✅ YES

## What It Does

This paper presents the first large-scale empirical analysis of arbitrage on Polymarket, analyzing **86 million transactions** across **17,218 conditions** from April 2024 to April 2025. It identifies and quantifies two distinct forms of arbitrage:

### 1. Market Rebalancing Arbitrage (Intra-Market)
- Occurs within a single market/condition
- Exploits YES + NO prices that don't sum to $1.00
- Found in **7,051 conditions** (41% of all markets analyzed)
- Estimated **$40M+ in extracted profits**

### 2. Combinatorial Arbitrage (Inter-Market)
- Spans across multiple logically related markets
- Exploits pricing inconsistencies between dependent conditions
- Uses a heuristic-driven reduction strategy to scale from O(2^(n+m)) to tractable analysis
- Validated by expert input and on-chain order book data

## Why It Matters

- **$40M in extracted profits** proves this is real, not theoretical
- **41% of markets** exhibit exploitable mispricings — massive surface area
- Provides the mathematical foundation for building automated arbitrage detection at scale
- The heuristic reduction strategy is implementable — reduces exponential complexity to linear-time filtering

## Architecture

```
On-Chain Data → Condition Extraction → Heuristic Filtering
                                        ↓
                              Timeliness Filter
                              Topical Similarity Filter
                              Combinatorial Relationship Filter
                                        ↓
                              Expert Validation
                                        ↓
                              Arbitrage Detection
                                        ↓
                              Execution Engine
```

## Key Findings

| Metric | Value |
|--------|-------|
| Conditions analyzed | 17,218 |
| Conditions with single-market arb | 7,051 (41%) |
| Total extracted profit | ~$40M USD |
| Arbitrage types | Market Rebalancing, Combinatorial |
| Data span | April 2024 - April 2025 |
| Transactions analyzed | 86M |

## Risks

- **Competition**: Bots have been exploiting these opportunities since at least 2024; edges may erode
- **Gas costs**: On Polygon, but still non-trivial for frequent small trades
- **Execution latency**: Opportunities are fleeting (sub-second for popular markets)
- **Liquidity constraints**: Shallow order books limit position sizes

## Implementability: 4/5

- The mathematical framework is well-specified
- Heuristic reduction makes it computationally tractable
- On-chain data is publicly available via Dune Analytics / Polymarket's API
- Main challenge: execution speed and competing with existing bots

## Next Steps

1. Implement the heuristic reduction algorithm in Python
2. Build a real-time scanner using Polymarket's CLOB API
3. Backtest on historical data to validate the $40M claim
4. Build a minimal execution engine with Polymarket's exchange API
5. Run paper trading for 2 weeks before going live
