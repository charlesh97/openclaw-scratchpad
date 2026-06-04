# Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets

**Source:** https://arxiv.org/abs/2508.03474  
**Recommendation:** YES ✅  
**Published:** August 5, 2025  

## Summary

The most comprehensive analysis of arbitrage on Polymarket to date. Analyzed 86 million on-chain transactions (April 2024 – April 2025) and discovered $40 million in arbitrage profits extracted. Identifies two distinct forms of arbitrage: Market Rebalancing and Combinatorial.

## Key Findings

1. **Market Rebalancing Arbitrage** — within a single market when YES + NO < $1.00
   - Found in 7,051 out of 17,218 conditions
   - $40M total arbitrage profits captured
   - Heavily bot-dominated by mid-2025

2. **Combinatorial Arbitrage** — across multiple logically dependent markets
   - Much rarer — 0.24% of total profits
   - Requires detecting dependencies between conditions (e.g., "BTC > $100K June" AND "BTC > $110K June")
   - Most LLM-detected dependencies (62%) fail to produce profitable arb

3. **Key Insight:** The median deviation of $0.60 means markets frequently deviate by 40% from efficiency — not close to efficient

## Why It Matters

- Most cited prediction market paper of 2025-2026
- Provides the mathematical framework for arbitrage detection via integer programming
- Shows combinatorial arb is underexploited (0.24% of profits despite ~40% of opportunities)
- Directly actionable: the integer programming approach can be implemented

## Risks

- Simple YES-NO bundle arb is already saturated by sub-100ms bots
- Combinatorial arb is mathematically complex and often unprofitable (62% failure rate)
- Paper is from Aug 2025 — market conditions have evolved significantly
- Dynamic fee model introduced 2026 changed profitability calculations

## Implementability: 3/5

Strong theoretical framework. The integer programming approach for combinatorial arb is implementable but complex. Simple bundle arb is too competitive.

## Status: REFERENCE
