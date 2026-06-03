# Prediction Market Maker — Paradigm Challenge Runner-Up

**Source:** https://github.com/octavi42/prediction-market-maker

## What It Does

A market-making strategy that placed **#2 in Paradigm's Prediction Market Challenge** (April 2026). 110 strategy iterations in 8 hours. Scored $41.09 mean edge per simulation.

The bot places passive limit orders on a simulated binary YES/NO prediction market. Edge comes from:
- **~60% monopoly regime** — when the bot is the only liquidity provider, it captures wider spreads
- **~40% normal regime** — standard spread capturing against informed flow

## Key Discoveries (from the author's write-up)
- **The monopoly regime** is the single insight worth more than 100 parameter tweaks
- **Sizing matters more than you think** — match expected retail order flow
- **Volatility-adjusted quote filtering** — when to quote and when to sit out
- **Inventory skew** prevents catastrophic losses (removing it = -$7 swing from $41 to $34)
- Position limit: max 33.3% of total capital at any given probability

## Why It Matters

This is an **educational goldmine** — the author shared every failed strategy iteration. The write-up covers quoting mechanics, adverse selection, inventory risk, and order sizing with concrete data. Directly applicable to building a Polymarket market-making bot.

## Risks
- Simulated environment — real Polymarket has fees, gas costs, and real counterparties
- FIFO LOB simulation doesn't capture Polygon block latency
- The single-market scenario doesn't test multi-market correlations

## Implementability: 3/5

No working Polymarket integration (it's a simulator submission), but the strategy logic can be directly ported. The monopoly regime detection is particularly valuable for real deployment.

## Next Steps
1. Port the strategy core to Python/Rust with real Polymarket CLOB API
2. Add monopoly regime detection to live market assessment
3. Implement volatility-adjusted quoting from the insights
4. Paper trade the strategy before live deployment
