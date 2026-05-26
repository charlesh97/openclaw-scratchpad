# Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets

**Source:** https://arxiv.org/abs/2508.03474  
**Authors:** Oriol Saguillo, Vajiheh Ghafouri, Lucianna Kiffer, Guillermo Suarez-Tangil (IMDEA Networks Institute)  
**Published:** August 2025 (revised May 2026)  
**Type:** Academic paper (AFT 2025)

## Summary

This paper presents the **first large-scale empirical analysis of arbitrage on Polymarket**, analyzing 86 million transactions across 17,218 conditions from April 2024 to April 2025. It identifies and measures two distinct forms of arbitrage:

1. **Market Rebalancing Arbitrage** (intra-market) — When YES + NO prices don't sum to $1.00 within a single market
2. **Combinatorial Arbitrage** (inter-market) — Arbitrage spanning multiple logically related markets

### Key Findings

- **7,051 conditions** had single-market arbitrage opportunities
- **~$40 million in total arbitrage profits** extracted over the study period
- Combinatorial arbitrage is rarer but larger per-opportunity
- Arbitrage persists due to slow human reaction times and fragmented liquidity
- Uses integer programming to detect combinatorial dependencies

## Why It Matters

- Empirically confirms that arbitrage exists at scale on Polymarket
- Quantifies total extractable value (~$40M/year)
- Provides a mathematical framework for detecting combinatorial arbitrage
- Directly applicable to building automated arbitrage strategies

## Implementability: 4/5

- The integer programming framework can be translated to Python
- Single-market arb detection is straightforward (YES+NO != $1)
- Combinatorial arb requires mapping condition dependencies
- Paper's methodology can guide implementation

## Next Steps
1. Translate the integer programming arb detection into our codebase
2. Implement single-market arb scanner first (lowest hanging fruit)
3. Build condition dependency graph for combinatorial detection
