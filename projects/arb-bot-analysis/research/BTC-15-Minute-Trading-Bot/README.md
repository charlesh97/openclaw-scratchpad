# Polymarket BTC 15-Minute Trading Bot

**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot
**Recommendation:** MEDIUM (queue)
**Implementability:** 3/5

## What It Does

A production-grade algorithmic trading bot specifically for Polymarket's 15-minute BTC price prediction markets. Uses a 7-phase architecture with multiple signal sources:

- **Spike Detection** — Sudden price movements on Coinbase/Binance
- **Sentiment Analysis** — Real-time news/news sentiment parsing
- **Price Divergence** — BTC price divergence detection across exchanges
- **Fusion Engine** — Weighted voting to combine signals
- **Self-Learning** — Automatically optimizes signal weights based on historical performance

## Architecture

7 phases: Input (external data) → Ingestion (unify & validate) → Nautilus Core → Signal Processors → Fusion Engine → Risk Management → Execution → Monitoring → Learning (feedback loop)

Built with **Nautilus Trader** framework, **Redis** for mode switching, **Grafana** dashboards, and **Prometheus** metrics.

## Why It Matters

- Most sophisticated signal processing of any bot found this cycle
- Production-grade monitoring (Grafana) — not a black box
- Self-learning signal weight optimization
- Auto-recovery from WebSocket disconnects

## Risks

- Python 3.14+ required (very new) — compatibility risk
- Requires running Redis + Grafana + Prometheus stack
- Heavy infrastructure footprint for what may be thin margins
- BTC 15-min markets on Polymarket have been targeted by dynamic fee changes

## Next Steps

Queue for deeper evaluation. Worth exploring the signal fusion approach — the spike + divergence detectors could be extracted and reused.

## Implementability Score: 3/5

Excellent architecture but heavy infra requirements. Best use is extracting the signal processing patterns.
