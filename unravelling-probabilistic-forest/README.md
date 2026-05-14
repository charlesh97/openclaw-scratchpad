# Unravelling the Probabilistic Forest — Arbitrage in Prediction Markets

**Source:** https://arxiv.org/abs/2508.03474  
**Published:** August 2025 (v1)  
**Authors:** Oriol Saguillo et al.  
**Published in:** AFT 2025 (LIPIcs)  
**Recommendation:** MEDIUM — Fundamental research, not directly implementable

## Summary

This paper presents the **first large-scale analysis of arbitrage on Polymarket**, identifying two distinct forms:

1. **Market Rebalancing Arbitrage** (Intra-market) — Exploiting YES + NO price sum ≠ $1.00 within a single market. Also includes multi-condition rebalancing within a single event.

2. **Combinatorial Arbitrage** (Inter-market) — Exploiting pricing inconsistencies across logically-related but independently-priced markets. For example, if "Team A wins game" and "Team A wins championship" markets imply conflicting probabilities.

**Key finding:** ~$40 million in arbitrage profits extracted from Polymarket between April 2024 and April 2025.

## Why It Matters

- Directly quantifies the total addressable arbitrage opportunity: **$40M/year** 
- Provides a rigorous taxonomy of arbitrage types — essential for designing detection algorithms
- Uses on-chain order book data at scale — validation methodology we can replicate
- Published in top-tier venue (AFT, LIPIcs) — peer-reviewed

## Key Insights

| Metric | Value |
|--------|-------|
| Total arbitrage profit (Apr '24–Apr '25) | ~$40M |
| Market Rebalancing share | Majority |
| Combinatorial Arbitrage share | Smaller but growing |
| Average arb window duration | Seconds to minutes |
| Competition level | High — bots dominate simple arb |

## Implementability: 2/5

- No code provided — this is a measurement/empirical paper
- The taxonomy is essential for thinking about arb strategies but doesn't provide an executable system
- Combinatorial arbitrage detection requires modeling event dependencies (complex)
- Cross-reference with Flashbots Collective post has some implementation hints

## Next Steps

1. Implement Market Rebalancing Arb scanner (YES+NO < $1.00) — straightforward, already partially done
2. Study combinatorial relationship detection methodology for multi-market arb
3. Use paper's measurement framework to evaluate our own arb detection performance
4. Read Flashbots Collective discussion for implementation insights
