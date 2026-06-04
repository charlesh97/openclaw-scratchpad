# Polymarket Rust ML + LLM Trading Bot

**Source:** https://github.com/skharchikov/polymarket-bot  
**Recommendation:** MEDIUM  
**Language:** Rust + Python (ML ensemble)  

## What It Does

A Cargo workspace with two independent Rust trading bots: an ML-driven trading bot (XGBoost ensemble + LLM consensus + Bayesian updating) and a copy-trading bot that mirrors top leaderboard traders. They share a common library, PostgreSQL database, and Python ML sidecar server.

## Architecture

```
Cargo Workspace
├── polymarket-common (shared lib: data models, Kelly sizing, feature engineering)
├── trading-bot (ML signals: XGBoost + LLM + Bayesian)
└── copy-trading-bot (mirror top traders)

External: Polymarket Gamma/CLOB API, WebSocket, News RSS, OpenAI
Infra: PostgreSQL, FastAPI model server, Prometheus, Grafana
```

## Key Features

- 29 market features (v5 feature engineering)
- Kelly Criterion position sizing
- Telegram notifications
- XGBoost ensemble for probability prediction
- LLM consensus via OpenAI API
- Bayesian updating on new information
- Historical data crawler for backtesting

## Risks

- Rust + Python dual-language = complex build/setup
- Research project only — not intended for production
- XGBoost models need ongoing retraining
- LLM API costs can eat profits
- Copy trading quality depends on leaderboard dynamics

## Implementability: 2/5

Interesting architecture but research-grade only. Complex multi-language setup, significant infra requirements, no live trading track record.

## Status: QUEUED
