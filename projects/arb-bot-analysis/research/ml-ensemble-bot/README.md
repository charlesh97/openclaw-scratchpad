# ML Ensemble + LLM Trading Bot

**Source:** https://github.com/skharchikov/polymarket-bot

## What It Does

A Cargo workspace (Rust) containing two independent trading bots backed by a shared library, Python ML ensemble, and PostgreSQL.

### Components
- **trading-bot**: ML-driven signals using XGBoost ensemble + LLM consensus + Bayesian updating
- **copy-trading-bot**: Mirrors trades from top leaderboard traders

### Architecture
- Rust core for low-latency trading
- Python FastAPI sidecar for ML model serving
- PostgreSQL for persistent storage
- Prometheus + Grafana for monitoring
- Telegram for notifications

### Signal Pipeline
1. Feature engineering (29 market features)
2. XGBoost ensemble predictions
3. LLM consensus from OpenAI API (news sentiment, market analysis)
4. Bayesian updating to combine signals
5. Kelly Criterion position sizing

## Implementability: 3/5

Complex dual-language architecture. The ML ensemble approach is powerful but requires significant infrastructure. The Rust core is impressive for latency but overkill for our initial deployment.

## Recommendation: MEDIUM

The ML+LLM signal fusion pipeline is the standout feature. Consider adapting the Python model server + signal combination logic while keeping our own execution layer.
