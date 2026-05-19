# BTC 15-Minute Signal Fusion Trading Bot

**Source:** [aulekator/Polymarket-BTC-15-Minute-Trading-Bot](https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot)
**Recommendation:** MEDIUM

## What It Does

A production-grade algorithmic trading bot for Polymarket's 15-minute BTC price prediction markets. Built with a **7-phase architecture** that fuses multiple signal sources:

- **Spike Detection** — Sudden price movements on Coinbase/Binance
- **Sentiment Analysis** — News and social media signals
- **Price Divergence** — Cross-exchange price differences

### Architecture
```
External Data (Coinbase, Binance, News, Solana)
    → Ingestion → Nautilus Trading Framework
    → Signal Processors (Spike, Sentiment, Divergence)
    → Fusion Engine (Weighted Voting)
    → Risk Management ($1 max/trade, 30% stop loss)
    → Execution (Polymarket Orders)
    → Monitoring (Grafana)
    → Self-Learning (Weight Optimization)
```

## Implementability: 3/5

- Well-structured Python code with clear separation of concerns
- Self-learning component is ambitious but adds complexity
- Requires Redis + Grafana infrastructure

## Risks
- 15-minute crypto markets have high variance
- Polymarket's dynamic fees (introduced 2026) may erode edges
- $1 max/trade limits potential upside in safe mode

## Next Steps
Evaluate signal fusion approach and adapt for longer-duration markets first.
