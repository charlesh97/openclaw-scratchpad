# Unravelling the Probabilistic Forest — TL;DR

| Aspect | Detail |
|--------|--------|
| **Title** | Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets |
| **Date** | August 5, 2025 |
| **Authors** | Oriol Saguillo et al. |
| **Link** | https://arxiv.org/abs/2508.03474 |

## In one sentence

The first empirical study proving that $40M+ was extracted from Polymarket via two specific types of arbitrage.

## Two types of arbitrage detected

| Type | Description | Profit Share |
|------|-------------|--------------|
| **Market Rebalancing Arb** | YES+NO < $1.00 within single market | ~80% of arb profit |
| **Combinatorial Arb** | Mispricing across logically linked markets | ~20% of arb profit |

## Why 62% of LLM-detected dependencies fail

A follow-up analysis (Navnoor Bawa) found that 62% of dependencies detected by LLMs between markets don't generate profit — the paper's finding is that real arb requires deep microstructure understanding, not surface-level correlation.

## Practical takeaway

Focus on **Market Rebalancing Arb** (YES+NO < $1) for higher probability, **Combinatorial Arb** for higher margins but more complexity.
