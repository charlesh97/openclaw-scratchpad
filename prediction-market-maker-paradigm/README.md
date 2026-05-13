# Prediction Market Maker — Paradigm Challenge (#2 Ranked)

**Source:** https://github.com/octavi42/prediction-market-maker

## What it does

A complete market-making strategy that placed **#2 in Paradigm's Prediction Market Challenge** (April 2026). The repo documents 110 iterations over 8 hours of competition, providing a detailed case study in prediction market microstructure — quoting mechanics, adverse selection, inventory risk, and order sizing.

## Key strategies covered

- **Spread capture:** Profit from bid-ask spread between uninformed flow and true probability
- **Inventory management:** Dynamic position sizing to avoid outsized risk
- **Adverse selection avoidance:** Identifying informed vs. uninformed order flow
- **Probability estimation:** Using market data to estimate true probabilities

## Why it matters

This is a **battle-tested** strategy from a competitive tournament setting. The repo is essentially a textbook on prediction market microstructure with working code. 110 iterations means the strategy was iteratively refined under live competition conditions.

## Implementability: 4/5

Complete codebase with competition-tested strategies. Requires adapting to Polymarket's specific API but the core logic is directly transferable.

## Risks

- Competition settings may differ from live Polymarket conditions
- Requires understanding of microstructure concepts
- Needs calibration for different market types (sports vs crypto vs politics)

## Next Steps

1. Review the 110 iterations to understand strategy evolution
2. Extract the final market-making algorithm
3. Adapt order sizing to Polymarket's fee structure
4. Backtest against historical Polymarket order book data
