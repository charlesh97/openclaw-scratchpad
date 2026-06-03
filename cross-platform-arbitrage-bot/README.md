# Cross-Platform Arbitrage Bot (Polymarket + Kalshi)

**Source:** https://github.com/ImMike/polymarket-arbitrage

## What It Does

A Python bot that watches **10,000+ markets** across Polymarket and Kalshi simultaneously, detecting:

1. **Cross-Platform Arbitrage** — Price differences for the same prediction on Polymarket vs Kalshi
2. **Bundle Arbitrage** — YES + NO prices that don't sum to $1.00
3. **Market Making** — Captures spreads with competitive bid/ask orders

Comes with a live web dashboard (real-time opportunities + bot activity), simulation mode for testing, and proper risk management (position limits, loss limits, kill switch).

## Why It Matters

This is the most practical, ready-to-run cross-platform arbitrage bot found today. The Polymarket-Kalshi arbitrage window is particularly compelling because:
- Polymarket and Kalshi often list identical binary markets (BTC price, election odds)
- Settlement timing differences + fragmented liquidity create persistent spread opportunities
- The author reports 99.6% win rate in simulation with configurable edge thresholds

## Key Features

| Feature | Detail |
|---------|--------|
| Markets Scanned | 5,000+ Polymarket + 5,000+ Kalshi |
| Arbitrage Types | Cross-platform + bundle + market making |
| Risk Mgmt | Position limits, max daily loss, kill switch |
| Dashboard | Real-time web UI (port 8000) |
| Data Modes | Real market data OR simulation |
| Fee Accounting | Gas costs + platform fees baked into edge calc |

## Risks
- **Latency:** Simple Python + REST polling — will lose races to HFT bots on short-term markets
- **Fee erosion:** Kalshi charges fees; Polymarket dynamic fees for 15-min crypto markets
- **API keys:** Requires both Polymarket and Kalshi credentials
- **Slippage:** The demo sim results (99.6%) almost certainly degrade in live trading

## Implementability: 4/5

Clean Python 3.10+ codebase, well-documented, active repo, MIT license. Fastest path from zero to running arb scanner. Simulation mode means zero-risk testing.

## Next Steps
1. Clone and install: `pip install -r requirements.txt`
2. Start with simulation mode: `data_mode: "simulation"`
3. Evaluate cross-platform gaps for ~1 week
4. Start with tiny position sizes and `min_edge: 0.01` (1%)
