# Polymarket Bot — ML Ensemble + LLM Signals

**Source:** https://github.com/skharchikov/polymarket-bot
**Recommendation:** YES (Top 2 for today's email)
**Implementability:** 3/5
**Last updated:** ~April 2026

## What it does

A sophisticated Cargo (Rust) workspace containing two independent trading bots for Polymarket, backed by a shared library, a Python ML ensemble, and PostgreSQL. The architecture is genuinely dual-purpose:

1. **trading-bot** — ML-driven signals using XGBoost ensemble + LLM consensus + Bayesian updating
2. **copy-trading-bot** — Mirrors trades from top leaderboard traders in real-time

### Key features:
- **XGBoost Ensemble**: 29 engineered features from market microstructure
- **LLM Consensus**: OpenAI API signals from news analysis — multiple LLM calls averaged
- **Bayesian Updating**: Posterior probability updates as new data arrives
- **Kelly Criterion**: Optimal position sizing based on estimated edge
- **Full Infrastructure**: PostgreSQL, Prometheus, Grafana, Telegram notifications
- **Rust Core**: High-performance execution, WebSocket feeds, order management

## Architecture

```
                     ┌──────────────────────┐
                     │   polymarket-common   │
                     │   (shared library)    │
                     └──────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                   │
     ┌────────▼────────┐  ┌────▼────────────┐
     │   trading-bot   │  │ copy-trading-bot │
     │ (Rust binary)   │  │ (Rust binary)    │
     └────────┬────────┘  └────────┬─────────┘
              │                     │
     ┌────────▼────────┐           │
     │ Model Server    │           │
     │ (FastAPI/Python)│           │
     │ XGBoost + LLM   │           │
     └────────┬────────┘           │
              │                     │
     ┌────────▼────────┐  ┌────────▼─────────┐
     │   PostgreSQL    │  │ Polymarket API   │
     │   + Prometheus  │  │ Gamma + CLOB     │
     └─────────────────┘  └──────────────────┘
```

## Why it matters

The ML ensemble + LLM consensus approach is cutting-edge — combining traditional ML (XGBoost on microstructure features) with LLM-based signals (news sentiment, event reasoning). The Bayesian updating adds a principled uncertainty framework. Rust execution layer ensures sub-millisecond latency.

## Risks

- **Complexity**: Requires Rust toolchain, Python ML stack, PostgreSQL — significant infrastructure overhead
- **LLM Costs**: OpenAI API calls add variable cost per prediction
- **Research-only**: Author explicitly states "not intended for production use"
- **Overfitting risk**: 29 features on limited market history could lead to overfitting

## Next Steps

1. Set up the Rust workspace and Python model server
2. Run backtests on historical Polymarket data
3. Replace OpenAI LLM with local model (Llama/DeepSeek) to eliminate API costs
4. Add additional feature engineering for order book imbalance
