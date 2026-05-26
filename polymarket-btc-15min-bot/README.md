# Polymarket BTC 15-Minute Trading Bot

**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot  
**Author:** aulekator  
**Language:** Python  
**License:** MIT  
**Status:** Active (published ~2 weeks ago)

## What It Does

A production-grade algorithmic trading bot specifically designed for Polymarket's 15-minute BTC price prediction markets ("Will BTC be above $X in 15 minutes?"). It uses a modular 7-phase architecture that combines multiple signal sources to make trading decisions.

### Core Architecture (7 Phases)

1. **Ingestion** — Unifies and validates data from Coinbase, Binance, news feeds, and Solana
2. **Nautilus Core** — Trading framework built on NautilusTrader
3. **Signal Processors** — Three distinct signal types:
   - *Spike Detection* — Catches abrupt price moves
   - *Sentiment Analysis* — News and social media sentiment
   - *Price Divergence* — Cross-exchange price divergence detection
4. **Fusion Engine** — Weighted voting across signal processors
5. **Risk Management** — $1 max per trade, 30% stop loss, 20% take profit
6. **Execution** — Polymarket CLOB order placement
7. **Monitoring + Learning** — Grafana dashboards, Prometheus metrics, and self-optimizing weight adjustment

## Why It Matters

- **Self-learning** — Automatically optimizes signal weights based on performance, meaning it adapts to changing market conditions without manual retuning
- **Multi-signal fusion** — Combines spike/sentiment/divergence signals, reducing false positives vs single-signal approaches
- **Production-grade** — Auto-recovery, WebSocket reconnection, rate limiting, paper trading mode
- **NautilusTrader foundation** — Battle-tested open-source algorithmic trading framework underneath

## Risks

- 15-minute markets are fast — latency matters. Dedicated Polygon RPC nodes recommended
- $1 max trade sizing limits upside but protects downside
- Signal reliability varies with market regime; sentiment signal can lag
- Polymarket dynamic fee changes could eat into edge

## Implementability: 4/5

- Well-structured, documented, tested code in Python
- Clear configuration and setup instructions
- Requires Redis + Grafana for full monitoring stack
- Need Polymarket API credentials and funded wallet

## Next Steps

1. Clone and run in paper trading mode to observe performance
2. Evaluate signal accuracy on historical BTC 15-min markets
3. Consider extending to ETH and SOL 15-min markets
4. Integrate with our own risk engine
