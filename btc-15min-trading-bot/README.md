# Polymarket BTC 15-Minute Trading Bot

**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot  
**Recommendation:** YES ✅  
**Language:** Python 3.14+  
**Architecture:** 7-phase pipeline  

## What It Does

A production-grade algorithmic trading bot for Polymarket's 15-minute BTC price prediction markets. Uses a modular 7-phase architecture combining multiple signal sources (spike detection, sentiment analysis, price divergence), professional risk management ($1 max per trade, stop-loss, take-profit), and self-learning capabilities that automatically optimize signal weights.

## Architecture (7-Phase)

```
Input (Coinbase, Binance, News, Solana) 
  → Ingestion (unify & validate) 
  → Nautilus Core (trading framework) 
  → Signal Processors (Spike, Sentiment, Divergence) 
  → Fusion Engine (weighted voting) 
  → Risk Management ($1 max, stop-loss) 
  → Execution (Polymarket orders) 
  → Monitoring (Grafana dashboard) 
  → Learning (weight optimization, feedback loop)
```

## Why It Matters

- **Most complete bot found** — covers the full pipeline from data ingestion to execution
- **Self-learning** — automatically optimizes signal weights from performance feedback
- **Professional risk mgmt** — $1 max/trade, 30% stop-loss, 20% take-profit defaults
- **Dual-mode** — paper trading (simulation) and live mode without restart
- **Auto-recovery** — WebSocket reconnection, rate limiting, data validation
- **Monitoring** — Grafana + Prometheus dashboards

## Risks

- Strategy dependent on BTC-15m market microstructure which Polymarket changes (dynamic fees introduced 2026)
- Uses Nautilus Trader framework — adds dependency complexity
- Paper trading ≠ live edge; slippage and latency in simulation differ significantly
- $1 max per trade limits upside; scaling needs careful risk review
- Requires Redis for mode switching (additional infra)

## Implementability: 4/5

Well-documented, Python-based, clear setup instructions. Moderate infra requirements (Redis, Grafana optional). Can run in paper mode immediately.

## Next Steps

1. Clone and run in simulation mode with paper data
2. Tune signal weights for current market regime
3. Validate against Polymarket's dynamic fee model
4. Consider scaling parameters for live deployment
