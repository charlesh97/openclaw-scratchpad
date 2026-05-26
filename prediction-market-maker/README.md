# Prediction Market Maker — Paradigm Hackathon #2

**Source:** https://github.com/octavi42/prediction-market-maker
**Recommendation:** ✅ YES — direct, actionable market-making strategy with proven edge

## What it does

This repo documents a complete market-making strategy that placed **#2 out of all submissions** in Paradigm's Prediction Market Challenge (April 2026). In 110 iterations over 8 hours, the strategy achieved a mean edge of **$41.09 per simulation** — just $1.23 behind first place.

### Core Insight: The Monopoly Regime

The strategy operates in two distinct regimes:

1. **Monopoly Regime (~60% of edge)**: When the competitor's bid or ask disappears (true probability near 0 or 1), the strategy becomes the *only liquidity provider*. Retail has no choice but to trade at posted prices. The bot posts enormous size at extreme prices — e.g., at prob=0.02, it posts 4,250 shares at 0.02¢, giving retail almost nothing for their YES shares while capturing massive edge.

2. **Normal Regime (~40% of edge)**: Volatility-adjusted quoting with careful sizing matched to expected retail order flow. The bot filters out quotes that would get swept by the arbitrageur and only provides liquidity where there's a positive expected edge.

### Key Discoveries

- **Sizing matters more than pricing**: Matching order size to expected retail flow (not just quoting the right price) was the single biggest performance lever
- **Volatility-adjusted quote filtering**: When to quote vs. when to sit out — avoiding adverse selection from informed flow
- **Inventory management via skew**: Removing skew from quotes caused a -$7 score swing

## Why it matters

This is a rare *open-source, competitive-level* market-making strategy with documented performance metrics. It provides a complete case study in prediction market microstructure: the mechanics of quoting, adverse selection, inventory risk, and optimal sizing. Directly applicable to Polymarket's CLOB.

## Risks
- **Simulated environment**: The hackathon simulated a simplified order book — real Polymarket dynamics differ
- **Competition regime**: Real markets have more than one competitor (vs. one static hidden ladder)
- **Latency dependency**: Monopoly regime is most profitable but only arises at extreme probabilities — may be rare in liquid markets

## Implementability: 3/5
Python code is clean and well-documented, but it targets a simulated environment. Adapting to live Polymarket requires integrating with Polymarket's CLOB API, real order book data, and real competition patterns.

## Next Steps
1. Study the monopoly regime logic — this is the most transferable insight
2. Adapt the strategy framework to Polymarket's CLOB API
3. Backtest on historical order book data from active 15-min markets
4. Deploy with conservative sizing in low-competition markets
