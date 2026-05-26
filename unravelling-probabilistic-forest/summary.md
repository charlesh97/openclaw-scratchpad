# Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets

**Source:** https://arxiv.org/abs/2508.03474
**Published:** August 2025 (LIPIcs AFT 2025)

## Summary

The first large-scale analysis of arbitrage on Polymarket, analyzing **86 million transactions** across **17,218 conditions** from April 2024 to April 2025.

### Key Findings

**Two distinct forms of arbitrage:**

1. **Market Rebalancing Arbitrage** (Intra-market): YES + NO < $1.00 within a single market
   - Found in **7,051 conditions** (41% of all analyzed)
   - $40M+ in arbitrage profits extracted during the study period

2. **Combinatorial Arbitrage** (Inter-market): Inconsistencies across logically related markets
   - Far less common — only **0.24% of total profits**
   - 62% of LLM-detected dependencies fail to generate arbitrage profit
   - Requires integer programming to detect at scale

### Key Numbers
- 86 million bets analyzed
- $40M+ total arbitrage profits
- 7,051 conditions had single-market arbitrage
- 17,218 total conditions analyzed
- Combinatorial arbitrage: tiny fraction of total profits

### Methodology
Used on-chain historical order book data + integer programming to detect and classify arbitrage patterns. The paper provides a formal framework for understanding when these opportunities exist and when traders actually execute them.

### Implications
- Single-market bundle arb is the dominant source of structured profits
- Combinatorial arb is theoretically interesting but practically limited by shallow liquidity
- Retail traders providing liquidity are the unwitting counterparties to these strategies

## Next Steps
1. Implement the integer programming framework from the paper for automated arb detection
2. Focus on single-market bundle arbitrage as the highest-ROI strategy
3. Monitor for combinatorial arb only in highly liquid, correlated markets
