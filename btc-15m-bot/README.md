# BTC 15-Minute Trading Bot

**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot

## What It Does

A production-grade algorithmic trading bot for Polymarket's 15-minute BTC price prediction markets. Built with a **7-phase architecture** combining multiple signal sources, professional risk management, and self-learning capabilities.

### Architecture (7-Phase)
1. **Input** — External data from Coinbase, Binance, News, Solana
2. **Ingestion** — Data unification and validation
3. **Nautilus Core** — Trading framework integration
4. **Signal Processing** — Spike detection, sentiment analysis, price divergence
5. **Fusion Engine** — Weighted voting from all signals
6. **Risk Management** — $1 max per trade, 30% stop loss, 20% take profit
7. **Execution & Monitoring** — Polymarket orders, Grafana dashboards

### Key Features
- Multi-signal intelligence (spike, sentiment, divergence detection)
- Self-learning weight optimization
- Dual-mode (simulation/live toggle)
- Auto-recovery, rate limiting, WebSocket reconnection

## Implementability: 4/5

Well-documented Python codebase. Requires Redis and Grafana infrastructure but the core logic is clean and modular. Most adaptable to our own signal sources.

## Recommendation: MEDIUM

Solid architecture but very specific to 15-min BTC markets. Good reference for the multi-signal fusion approach.
