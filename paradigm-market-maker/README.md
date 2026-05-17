# Paradigm Prediction Market Maker (octavi42)

**Source:** https://github.com/octavi42/prediction-market-maker

## What It Does
A market-making strategy that placed **#2 in Paradigm's Automated Research Hackathon** (April 9, 2026). The strategy was refined across 110 iterations in 8 hours, achieving a mean edge of $41.09 per simulation — just $1.23 behind the winner.

The challenge: build a market-making strategy for a simulated binary YES/NO prediction market with a FIFO limit order book, where the strategy can only place passive (limit) orders.

## Why It Matters
This is the first publicly documented, benchmarked strategy from a structured prediction market-making competition. Key insights from 110 iterations:

- **~60% of edge comes from identifying and exploiting a "monopoly regime"** — when the competitor agent's ladder is thin, you can dominate the spread with wide, high-volume quotes
- **The other ~40% comes from normal regime quoting** — volatility-adjusted quote filtering, intelligent sizing relative to expected retail order flow
- **Inventory management is critical** — removing skew from the strategy costs -$7 in edge (17% of total)
- **Quote filtering beats brute force** — sitting out high-volatility periods prevents adverse selection by the arbitrageur agent

## Architecture
- Python-based simulation engine
- Strategy parameters optimized via 110 iterative runs
- Competitor model: static hidden ladder, replenishes consumed levels
- Arbitrageur model: knows true probability, sweeps mispriced orders
- Retail model: random market orders (~0.25/step, ~$4.5 mean notional)

## Risks
- Simulated on stylized order book — real Polymarket microstructure differs significantly (gas costs, latency, multiple competitors)
- The "monopoly regime" insight may not transfer to live markets with dozens of competing MMs
- Hackathon time horizon (8 hours) doesn't cover multi-week strategy drift

## Implementability: 5/5
**YES for email.** Complete open-source codebase, well-documented strategy analysis, Python implementation. The core insights (volatility-adjusted quoting, inventory skew, regime detection) are directly implementable on Polymarket via the CLOB API. Start with the monopoly regime detection as a paper-trading module.

## Next Steps
1. Clone the repo and reproduce the simulation results
2. Port the regime detection logic to live Polymarket data
3. Replace the stylized order book with Polymarket CLOB snapshots
4. Add gas cost model to the P&L simulation
5. Paper trade the monopoly regime strategy on 15-min BTC markets
