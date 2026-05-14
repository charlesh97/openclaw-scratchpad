# Polymarket-bot — 4-Strategy Polymarket Trading Bot

**Source:** https://github.com/MrFadiAi/Polymarket-bot  
**Status:** Actively maintained (v3.1 — January 2026)  
**Language:** Python  
**Recommendation:** ✅ YES — Top Pick for Email

## What It Does

A unified trading bot that implements 4 distinct strategies for Polymarket prediction markets, with an auto-rotation system that switches between strategies based on real-time performance:

1. **Smart Money Strategy** — Tracks and mirrors trades from identified "smart" wallets (whales, profitable traders)
2. **Momentum Strategy** — Detects short-term price momentum and streaks
3. **Mean Reversion Strategy** — Bets against extreme pricing when probability diverges significantly from modeled fair value
4. **Spread Arbitrage Strategy** — Captures YES+NO < $1.00 bundle arbitrages across related markets

## Why It Matters

- **Multi-strategy rotation** handles different market regimes automatically — a sophisticated approach beyond single-strategy bots
- v3.1 (Jan 2026) includes dynamic position sizing and enhanced risk management
- Production-tested with dashboard UI and auto-rotation
- Open source with 4 battle-tested strategies that can be studied individually and combined

## Architecture

```
Polymarket-bot/
├── strategies/
│   ├── smart_money.py    — Whale/copy-trading detection
│   ├── momentum.py       — Trend-following signals
│   ├── mean_reversion.py — Statistical mispricing recovery
│   └── spread_arb.py     — Bundle arbitrage detection
├── core/
│   ├── risk_manager.py   — Dynamic sizing, stop-loss, drawdown control
│   ├── market_data.py    — Order book + trade feed ingestion
│   └── executor.py       — CLOB API execution wrapper
├── dashboard/            — Real-time performance UI
└── config.yaml           — Strategy weights, thresholds, capital allocation
```

## Risks

- **Strategy overlap risk:** Auto-rotation may allocate capital to temporarily-correlated strategies, amplifying drawdowns
- **Latency dependency:** Spread arbitrage requires fast execution (sub-500ms) to capture fleeting YES+NO dislocations
- **Smart Money strategy degrades** if whales switch wallets or use obfuscation
- Platform-level changes (fee structure, API changes) can break assumptions across all 4 strategies simultaneously

## Implementability: 4/5

- Python-based, well-documented, clean separation of concerns
- Requires Polymarket API key setup and Polygon RPC endpoint
- Dashboard dependency adds deployment complexity but is optional
- Can extract individual strategy modules for targeted backtesting

## Next Steps

1. Clone and run dry-mode on historical data for each strategy independently
2. Backtest spread arbitrage strategy on 15-minute BTC markets (highest liquidity)
3. Compare mean reversion signals against our own CLOB data pipeline
4. Evaluate whether auto-rotation adds alpha vs. static strategy allocation
5. Port relevant risk management logic (dynamic sizing, drawdown limits) into our bot
