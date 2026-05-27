# Paradigm Prediction Market Maker

**Source:** https://github.com/octavi42/prediction-market-maker

## What It Does

A market-making strategy that placed **#2 in Paradigm's Prediction Market Challenge** (April 9, 2026 hackathon). The competition tasked participants with building an automated market-making strategy for binary YES/NO prediction markets with a FIFO limit order book.

The strategy achieved a $41.09 mean edge per simulation across 110 iterations developed in 8 hours — just $1.23 behind the #1 entry.

## Why It Matters

This is the closest open-source proxy to a production-grade market-making algorithm for Polymarket-style prediction markets. The repo is a complete case study covering:

- **Mechanics of quoting** — how to price limit orders relative to true probability
- **Adverse selection** — how the arbitrageur agent punishes bad quotes
- **Inventory risk** — skew management prevents catastrophic losses
- **Order sizing** — matching expected retail order flow

### Key Discovery: The "Monopoly Regime"

~60% of edge came from a single insight: when the true probability is near 0 or 1 (extreme beliefs), most other market participants are on one side of the book, creating a monopoly opportunity for the market maker on the other side. The remaining ~40% came from normal regime trading.

## Risks

- **Simulated environment only** — tested against stylized agents, not real Polymarket order books
- **No real liquidity constraints** — the simulation had continuous replenishment
- **No gas/fee model** — Polymarket's dynamic taker fees (introduced 2026) would erode margins
- **No multi-market exposure** — single binary market only

## Implementability: 4/5

The strategy logic is straightforward to port to Polymarket's CLOB. Key components:
- Probability estimation (use CLOB mid-price + external signals)
- Volatility-adjusted quoting
- Inventory skew management
- Position sizing via Kelly Criterion

Needs adaptation for multi-market, multi-wallet deployment.

## Next Steps

1. Port the quoting logic to Polymarket's CLOB API in Python
2. Replace the simulation's "true probability" with a composite signal (order book + external data)
3. Add multi-market inventory management
4. Backtest against historical order book data (see polymarket-microstructure research)
5. Implement the "monopoly regime" detection as a standalone module
