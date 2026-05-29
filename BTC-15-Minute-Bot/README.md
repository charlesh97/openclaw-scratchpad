# Polymarket BTC 15-Minute Trading Bot

**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot
**Recommendation:** YES (Top 2 for today's email)
**Implementability:** 4/5
**Last updated:** ~May 2026

## What it does

A production-grade algorithmic trading bot for Polymarket's 15-minute BTC price prediction markets. Uses a **7-phase modular architecture** that ingests data from multiple sources (Coinbase, Binance, Solana, news feeds) and fuses them through a weighted voting system.

### Key features:
- **Multi-Signal Intelligence**: Spike detection, sentiment analysis, and price divergence indicators run in parallel
- **Fusion Engine**: Weighted voting system that combines signals, with self-learning weight optimization
- **Risk-First Design**: $1 max per trade, 30% stop loss, 20% take profit — extremely disciplined
- **Dual-Mode Operation**: Toggle between simulation and live without restarting
- **Real-Time Monitoring**: Grafana dashboards + Prometheus metrics
- **Auto-Recovery**: WebSocket auto-reconnection, rate limiting, data validation

## Architecture

```
External Data (Coinbase, Binance, News, Solana)
  → Ingestion (Unify & Validate)
    → Nautilus Core (Trading Framework)
      → Signal Processors (Spike, Sentiment, Divergence)
        → Fusion Engine (Weighted Voting)
          → Risk Management ($1 Max, Stop Loss)
            → Execution (Polymarket Orders)
              → Monitoring (Grafana Dashboard)
                → Learning (Weight Optimization)
```

## Why it matters

This is one of the few genuinely production-grade open-source Polymarket bots. The 7-phase architecture with self-learning weight optimization makes it adaptable to changing market regimes. The risk management is professional-grade ($1 max per trade is extremely conservative — great for validation).

## Risks

- Requires Python 3.14+, Redis, and Polymarket API credentials
- 15-minute BTC markets are highly competitive (bots dominate)
- Self-learning could overfit to recent market conditions
- Documentation incomplete for some modules

## Next Steps

1. Clone and run in simulation mode first
2. Tune signal weights for our risk preferences
3. Add additional data sources (on-chain metrics, order book imbalance)
4. Extend to 1-hour BTC and ETH markets
