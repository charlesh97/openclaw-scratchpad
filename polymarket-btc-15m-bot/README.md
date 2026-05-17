# Polymarket BTC 15-Minute Trading Bot (aulekator)

**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot

## What It Does
A production-grade algorithmic trading bot for Polymarket's 15-minute BTC price prediction markets. Uses a **7-phase architecture** combining multiple signal sources, professional risk management, and self-learning capabilities.

## Architecture (7-Phase)

```
External Data (Coinbase, Binance, News, Solana)
    ↓
Ingestion → Unify & Validate
    ↓
Nautilus Core (Trading Framework)
    ↓
Signal Processors (Spike, Sentiment, Divergence)
    ↓
Fusion Engine (Weighted Voting)
    ↓
Risk Management ($1 Max, Stop Loss)
    ↓
Execution (Polymarket Orders)
    ↓
Monitoring (Grafana Dashboard) → Learning (Weight Optimization)
    ↓
Feedback loop back to Fusion Engine
```

## Why It Matters
- **Multi-Signal Intelligence:** Spike Detection + Sentiment Analysis + Price Divergence — fused via weighted voting
- **Risk-First Design:** $1 max per trade, 30% stop loss, 20% take profit
- **Self-Learning:** Automatically optimizes signal weights based on performance
- **Dual-Mode Operation:** Toggle between simulation and live without restart
- **Auto-Recovery:** WebSocket auto-reconnection, rate limiting, data validation
- **Full monitoring stack:** Grafana dashboards + Prometheus metrics

## Risks
- 15-min BTC markets are the most competitive on Polymarket (latency arb bots, dynamic fees)
- Relies on centralized data sources (Coinbase, Binance) — API failures cascade
- Self-learning adds overfitting risk in regime-changing markets

## Implementability: 5/5
**YES for email.** Full production setup with Docker, Nautilus framework, Grafana dashboards. Python 3.14+. The architecture is modular — each signal processor can be independently developed, tested, and deployed. Self-learning weight optimization is a genuine edge.

## Next Steps
1. Deploy the bot in simulation mode on 15-min BTC markets
2. Add custom signal processors for on-chain Polymarket data
3. Extend to ETH, SOL 15-min markets
4. Add cross-market signal fusion (BTC signal → ETH market)
5. Implement the self-learning weight optimizer on a 7-day rolling window
