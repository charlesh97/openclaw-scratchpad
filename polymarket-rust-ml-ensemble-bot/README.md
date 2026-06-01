# Polymarket Rust ML Ensemble Bot

**Source:** https://github.com/skharchikov/polymarket-bot
**Recommendation:** YES — Top priority for study and integration
**Implementability:** 5/5

---

## What It Does

A production-grade Rust + Python ML ensemble trading bot for Polymarket prediction markets. Ships two independent bots in a single Cargo workspace:

1. **trading-bot** — ML-driven signal generation using XGBoost ensemble + LLM consensus + Bayesian updating
2. **copy-trading-bot** — Mirrors trades from top leaderboard traders in real-time

Both bots share a common library, PostgreSQL database, and Prometheus/Grafana monitoring infrastructure.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Cargo Workspace                     │
│                                                   │
│  ┌─────────────┐    ┌──────────────────┐        │
│  │ trading-bot │    │ copy-trading-bot │        │
│  │  (binary)   │    │    (binary)      │        │
│  └──────┬──────┘    └────────┬─────────┘        │
│         │                   │                    │
│         └────────┬──────────┘                    │
│                  ▼                               │
│         ┌────────────────┐                       │
│         │  polymarket-   │                       │
│         │  common (lib)  │                       │
│         └────────────────┘                       │
│                                                   │
│  External Services:                               │
│  - Gamma API (market metadata)                   │
│  - CLOB API (order execution)                    │
│  - WebSocket (real-time price data)              │
│  - News RSS Feeds                                │
│  - OpenAI API (LLM consensus)                    │
│                                                   │
│  Infrastructure:                                  │
│  - PostgreSQL (persistence)                      │
│  - Python FastAPI sidecar (model server)         │
│  - Prometheus + Grafana (monitoring)             │
└─────────────────────────────────────────────────┘
```

## Signal Pipeline (trading-bot)

1. **Gamma Scanner** — Fetches live markets from Polymarket Gamma API
2. **Feature Engineering** — 29 features (MarketFeatures v5) covering price, volume, order book, time-based signals
3. **XGBoost Ensemble** — ML predictions from the Python model server (FastAPI sidecar)
4. **LLM Consensus** — Multi-source news → embedding similarity → OpenAI prompt → probability estimate
5. **Bayesian Updating** — Combines XGBoost and LLM signals using likelihood ratios, calibrated from resolved markets
6. **Kelly Criterion** — Position sizing based on confidence and bankroll
7. **Execution** — Order placement via CLOB API
8. **Housekeeping** — Stop-loss, expiry exits, recalibration

## Key Features

| Feature | Details |
|---------|---------|
| **Language** | Rust (core) + Python (ML sidecar) |
| **ML Model** | XGBoost ensemble with 29 engineered features |
| **LLM Integration** | OpenAI API for consensus probability estimation |
| **Position Sizing** | Full Kelly Criterion with configurable risk profiles |
| **Risk Profiles** | Aggressive / Balanced / Conservative |
| **Monitoring** | Prometheus metrics + Grafana dashboards |
| **Alerting** | Telegram bot with subscriber management |
| **Backtesting** | Built-in historical data crawler |
| **Stop-Loss** | Automatic exit on adverse moves |

## Signal Sources

- `SignalSource::XgBoost` — ML ensemble prediction
- `SignalSource::LlmConsensus` — LLM-backed probability estimate
- `SignalSource::CopyTrade` — Mirror from top traders
- `SignalSource::Bayesian` — Combined Bayesian-updated signal

## Why This Matters

This is the most complete open-source ML prediction market bot found to date. It has:

- **Real ML** — Not just hard-coded rules; actual XGBoost ensemble with feature engineering
- **LLM integration** — Novel use of LLMs as a signal source alongside traditional ML
- **Bayesian framework** — Clean mathematical approach to combining disparate signals
- **Full infrastructure** — Monitoring, persistence, alerting, backtesting
- **Two strategies** — ML-driven and copy-trading in one codebase

## Risks

- Research project only — explicitly not production-ready
- Requires OpenAI API access (ongoing cost)
- Polygon RPC node needed for reliable execution
- Rust compilation (+ ML dependencies) require setup

## Next Steps

1. Clone the repo and build the Cargo workspace
2. Set up PostgreSQL + FastAPI model server
3. Run in paper/simulation mode for 2+ weeks
4. Evaluate signal quality vs. baseline strategies
5. Port the ML ensemble module to our arb-bot-analysis Python framework

## Integration Potential

The `polymarket-common` shared library could be extracted and used as a Rust SDK for our own bots. The feature engineering (29 features in `MarketFeatures`) is directly applicable to any prediction market strategy. The Bayesian combining framework is architecture-agnostic — we could implement it in Python.
