# BTC 15-Minute Trading Bot

**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot

## What It Does

A production-grade algorithmic trading bot specifically for Polymarket's **15-minute BTC price prediction markets**. Built with a 7-phase architecture:

1. **Ingestion** — Unify & validate data from Coinbase, Binance, News feeds, Solana
2. **Nautilus Core** — Trading framework
3. **Signal Processors** — Spike Detection, Sentiment Analysis, Price Divergence
4. **Fusion Engine** — Weighted voting across signals
5. **Risk Management** — $1 max per trade, 30% stop loss, 20% take profit
6. **Execution** — Polymarket order routing
7. **Self-Learning** — Automatic weight optimization based on performance

## Key Features
- **Multi-signal intelligence** (spike, sentiment, divergence)
- **Dual-mode** — toggle between simulation and live without restart
- **Grafana dashboards** + Prometheus metrics
- **WebSocket auto-reconnection** + rate limiting
- **Paper trading** with full P&L tracking

## Risks
- Narrow focus: only BTC 15-minute markets
- Polymarket's new **dynamic taker fees** directly target this strategy type
- Multi-signal latency may miss short windows
- Self-learning adds complexity without proven edge

## Implementability: 3/5

Good documentation and architecture, but the narrow BTC 15-min focus and dynamic fee headwind reduce applicability. Architecture patterns (multi-signal fusion, risk framework) are worth extracting.

## Next Steps
1. Test in simulation mode to evaluate signal quality
2. Assess impact of Polymarket's dynamic fee structure
3. Consider expanding architecture to multi-market
