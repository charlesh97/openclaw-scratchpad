# Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets

**Source:** https://arxiv.org/abs/2508.03474

## What it does
This paper presents the first large-scale empirical analysis of arbitrage on Polymarket. It identifies and characterizes **two distinct forms of arbitrage**:

1. **Market Rebalancing Arbitrage** — Occurs within a single market (e.g., YES + NO prices < $1.00). The classic "risk-free" arb where you buy both sides for < $1 and collect the difference at settlement.

2. **Combinatorial Arbitrage** — Spans logically related markets. Example: if "Candidate X wins Party A primary" and "Candidate X wins general election" are priced inconsistently with the joint probability constraints, an arb exists across both markets.

Uses on-chain historical order book data to measure when these opportunities existed and whether they were exploited.

## Why it matters
- **Directly relevant to our arb bot strategy** — this paper gives us the mathematical framework for detecting combinatorial arb.
- **Quantifies real opportunity sizes** — tells us what spreads actually existed, for how long, and at what volume.
- **Validates the approach** — proves that systematic arb in prediction markets is both detectable and profitable.

## Key Findings
- Both arb types exist persistently across Polymarket markets
- Combinatorial arb opportunities are more complex to detect but can have larger theoretical edge
- Most arb opportunities decay within minutes as bots and market participants compete
- Market Rebalancing arb (YES+NO < $1) is the most accessible for automated detection

## Implementability: 3/5
The paper provides the theoretical framework but no code. The combinatorial arb detection requires building a graph of logically related markets and solving constraint satisfaction problems. Significant engineering effort.

## Next Steps
1. Implement Market Rebalancing arb detection (simpler, immediate payoff)
2. Build a graph of Polymarket events and their logical dependencies
3. Implement combinatorial arb constraint solver (integer programming)
4. Backtest against historical order book data
