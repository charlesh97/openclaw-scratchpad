# Unravelling the Probabilistic Forest — Arbitrage in Prediction Markets

**Source:** https://arxiv.org/abs/2508.03474  
**Recommendation:** YES

## Summary

The first large-scale empirical analysis of arbitrage on Polymarket, analyzing 86 million transactions from April 2024 to April 2025. Documents $40M+ in extracted arbitrage profits across two distinct arbitrage forms.

## Two Types of Arbitrage Identified

### 1. Market Rebalancing Arbitrage (Intra-Market)
Occurs within a single market. YES + NO prices not summing to $1.00. The classic "free money" opportunity. Most common in newly created or volatile markets.

### 2. Combinatorial Arbitrage (Inter-Market)
Spans across multiple related markets. Exploits pricing inconsistencies between logically dependent conditions. Requires solving O(2^n) comparisons, reduced via heuristic-driven strategy.

## Key Numbers

- 17,218 total conditions analyzed
- 7,051 conditions had single-market arbitrage
- $40M+ realized profit extracted
- 40% median deviation ($0.60) — market frequently deviates significantly from efficiency

## Methodological Contribution

Uses integer programming to detect arbitrage at scale. The heuristic reduction strategy based on timeliness, topical similarity, and combinatorial relationships makes the problem tractable, validated by expert input.

## Why It Matters

This is now the foundational empirical reference for Polymarket arbitrage. The $40M+ figure sets a floor on the total arbitrage opportunity. The methodology can be replicated and extended to build a live arbitrage detection system.

## Implementability: 3/5

The methodology is well-documented but implementing the integer programming solution for live detection requires significant development. However, the simpler market rebalancing arbitrage (YES+NO < $1) is straightforward to implement.

## Next Steps

- Implement the integer programming arbitrage detection
- Build a live scanner for market rebalancing arbitrage
- Extend analysis to combinatorial arb across related events
