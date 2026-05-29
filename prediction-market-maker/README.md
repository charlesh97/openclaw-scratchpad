# Prediction Market Maker — #2 in Paradigm Challenge

**Source:** https://github.com/octavi42/prediction-market-maker
**Recommendation:** MEDIUM
**Implementability:** 4/5
**Last updated:** April 2026

## What it does

A market-making strategy that placed **#2 out of all submissions** in Paradigm's Prediction Market Challenge (April 2026). Built in 8 hours with 110 iterations. Score: $41.09 mean edge per simulation (winner: $42.32).

### Key discoveries:
- **The monopoly regime**: Strategy earned ~60% of edge when it was the sole liquidity provider on one side of the book
- **Sizing matters more than parameters**: Matching expected retail order flow was the critical lever
- **Volatility-adjusted quoting**: When to quote and when to sit out based on recent variance
- **Inventory skew**: Removing inventory management caused a -$7 swing in performance

## Why it matters

This is a complete case study in prediction market microstructure. The insights about monopoly regime (being the only liquidity provider on a side), order sizing to match retail flow, and volatility filtering are directly applicable to our bots.

## Risks
- Simulated environment (FIFO LOB, integer ticks 1-99) differs from real Polymarket CLOB
- Competitor behaviors are stylized, not real
- Requires adaptation to real market conditions

## Next Steps
1. Study the 110 strategy iterations for progression insights
2. Implement the monopoly-regime detection in our bots
3. Build a similar simulation environment for testing
