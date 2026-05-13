# Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets

**Source:** https://arxiv.org/abs/2508.03474

## What it does

The first large-scale analysis of arbitrage on Polymarket. Quantifies **$40M USD of profit extracted** via arbitrage strategies. Reveals two distinct forms of arbitrage: Market Rebalancing Arbitrage (intra-market) and Combinatorial Arbitrage (inter-market).

## Key findings

- **Market Rebalancing Arbitrage:** Within a single market — YES + NO prices summing to < $1.00
- **Combinatorial Arbitrage:** Across multiple related markets — pricing inconsistencies between logically linked events
- **$40M extracted profit:** Realized arbitrage profits between April 2024 and April 2025
- **On-chain analysis:** Used historical order book data from Polymarket

## Why it matters

This is the **definitive empirical study** of arb on Polymarket. The $40M figure validates that arb is a real, profitable strategy. Understanding the two types (intra-market vs. combinatorial) is essential for building a profitable bot.

## Implementability: 3/5

Research paper with detailed methodology but no code. The algorithmic approach can be reconstructed from the paper's methodology section.

## Risks

- Most simple arb (YES+NO < $1) is now captured by HFT bots
- Combinatorial arb requires understanding complex market relationships
- Paper covers 2024-2025 data; the arb landscape has evolved

## Next Steps

1. Read the full paper for detection methodology
2. Replicate the Market Rebalancing Arb detection algorithm
3. Build Combinatorial Arb detection for sports/election markets
4. Backtest on current Polymarket data to verify persistence
