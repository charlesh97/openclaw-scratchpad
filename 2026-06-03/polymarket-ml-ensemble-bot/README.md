# Polymarket ML Ensemble Bot (polybase)

**Source:** https://github.com/skharchikov/polymarket-bot

## What It Does

A Cargo workspace with two independent Rust trading bots for Polymarket, backed by a shared library, a Python ML ensemble sidecar (FastAPI), and PostgreSQL:

1. **trading-bot** — ML-driven signals combining XGBoost ensemble, LLM consensus (OpenAI), and Bayesian updating
2. **copy-trading-bot** — Mirrors trades from top leaderboard traders

Both bots share a common library, database, and monitoring stack (Prometheus + Grafana). Telegram interface for each.

## Why It Matters

This is one of the most technically sophisticated open-source Polymarket bots available. The hybrid ML ensemble architecture (XGBoost + LLM + Bayesian) is genuinely novel and represents the cutting edge of prediction market signal processing. The dual-bot architecture lets you run both directional strategies and copy-trading simultaneously.

## Architecture

```
                    ┌─ Polymarket Gamma API ─┐
                    │ Polymarket CLOB API    │
                    │ WebSocket Feeds        │
                    │ News RSS               │
                    │ OpenAI API             │
                    └────────┬───────────────┘
                             │
              ┌──────────────┴──────────────┐
              │       Cargo Workspace       │
              │  ┌──────────────────────┐  │
              │  │  polymarket-common   │  │
              │  │   (shared library)   │  │
              │  └──────┬───────┬───────┘  │
              │  ┌──────┘       └───────┐  │
              │  ▼                      ▼  │
              │ trading-bot    copy-trading │
              │ (ML signals)   (mirror)     │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │     Infrastructure          │
              │  PostgreSQL | Model Server  │
              │  Prometheus | Grafana       │
              └─────────────────────────────┘
```

## Signal Stack
- **XGBoost ensemble** — trained on market microstructure data
- **LLM consensus** — OpenAI calls for narrative/event interpretation
- **Bayesian updating** — combines priors with streaming evidence

## Risks
- Significant infrastructure complexity (Rust + Python + PostgreSQL + monitoring)
- LLM latency may be too high for fast markets
- Requires Polymarket API keys and OpenAI API key
- Research-only project declared by author — not production-ready

## Implementability: 4/5

Well-documented, active repo. Rust and Python skills required but architecture is clean and modular. Docker Compose setup streamlines deployment.

## Next Steps
1. Clone and set up with Docker Compose
2. Configure API keys in `.env`
3. Start with paper trading mode
4. Evaluate ML ensemble performance vs. simple strategies
