# Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets

**Source:** https://arxiv.org/abs/2508.03474 (August 2025)

## Summary

The most comprehensive analysis of arbitrage on Polymarket to date. The researchers analyzed **86 million transactions** across the full Polymarket order book from April 2024 to April 2025.

### Key Findings

1. **$40 million in arbitrage profits extracted** during the study period
2. **Two distinct arbitrage forms identified:**
   - **Market Rebalancing Arbitrage** — within a single market or condition (YES + NO < $1.00)
   - **Combinatorial Arbitrage** — spanning across multiple related markets
3. **7,051 out of 17,218 conditions had single-market arbitrage** opportunities
4. **Combinatorial arbitrage** is rarer but yields larger profits when found
5. Used integer programming to detect arbitrage at scale across 17,218 conditions

### Methodology
- Heuristic-driven reduction strategy based on timeliness, topical similarity, and combinatorial structure
- Overcame O(2^(n+m)) complexity through intelligent pruning
- On-chain historical order book data

### Relevance to Our Bot
Directly quantifies the arbitrage opportunity space. The combinatorial arbitrage detection methodology is implementable as a scanner module. Establishes that arb profits existed at scale, though competition has increased dramatically since April 2025.

## Implementability: 3/5
The methodology is clear but requires significant data infrastructure. The combinatorial detection algorithm (integer programming over condition graphs) is the most valuable — could be built as a standalone scanner.
