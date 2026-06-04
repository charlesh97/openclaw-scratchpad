## 2026-06-04 Research Findings

### 1. BTC 15-Minute Trading Bot
**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot
**Recommendation:** YES

**What it does:**
- Production-grade algorithmic trading bot for Polymarket's 15-minute BTC prediction markets
- 7-phase architecture: data ingestion → signal processing (spike/sentiment/divergence) → fusion engine → risk management → execution → monitoring → self-learning
- Multi-signal intelligence combining market data, sentiment analysis, and price divergence
- Self-learning capabilities that automatically optimize signal weights
- Dual-mode operation (simulation/live), paper trading, auto-recovery

**Implementability:** 4/5 — Well-documented Python, clear setup, moderate infra (Redis). Can paper trade immediately.

---

### 2. Polymarket-Kalshi Cross-Platform Arbitrage Bot
**Source:** https://github.com/ImMike/polymarket-arbitrage
**Recommendation:** YES

**What it does:**
- Scans 10,000+ markets across Polymarket and Kalshi for arbitrage opportunities
- Three strategies: cross-platform arb, bundle arb (YES+NO < $1), market-making
- Real-time web dashboard, fee accounting, NLP market matching
- Simulation mode (99.6% win rate) and live mode
- Complete risk management with kill switch

**Implementability:** 5/5 — Best documentation found. Clear config, one-command setup. Perfect starting point.

---

### 3. Unravelling the Probabilistic Forest (ArXiv)
**Source:** https://arxiv.org/abs/2508.03474
**Recommendation:** YES (reference paper)

**What it does:**
- Analyzed 86M Polymarket transactions over 12 months
- Identified $40M in arbitrage profits extracted
- Two types: Market Rebalancing (YES+NO < $1) and Combinatorial (cross-market dependency)
- Market Rebalancing is now HFT-bot saturated; Combinatorial arb is 0.24% of profits but underexploited
- Integer programming framework for detecting dependencies

**Implementability:** 3/5 — Theoretical framework. Combinatorial arb is complex but promising.

---

### 4. Polymarket Microstructure Anatomy (ArXiv)
**Source:** https://arxiv.org/abs/2604.24366
**Recommendation:** MEDIUM (reference paper)

**What it does:**
- 30 billion order-book events analyzed over 52 days across 600 markets
- 8 stylized facts about Polymarket microstructure
- Sub-50ms median ingestion delay, uniform depth profile, 1% median wash share
- Essential context for building and calibrating any Polymarket strategy

**Implementability:** 3/5 — Academic research. Provides essential empirical context.

---

### 5. NBA Arbitrage Analysis (ArXiv)
**Source:** https://arxiv.org/abs/2605.00864
**Recommendation:** NO (too constrained)

**What it does:**
- Sports arbitrage analysis in Polymarket NBA markets
- 290 Moneyline-Spread arb episodes found
- 76.9% constrained to 14.8 shares average — liquidity severely limits value
- Only viable at retail scale

**Implementability:** 2/5 — Liquidity-constrained. Not actionable at meaningful scale.

---

### 6. Polymarket Bot — 4 Strategies (MrFadiAi)
**Source:** https://github.com/MrFadiAi/Polymarket-bot
**Recommendation:** MEDIUM (queued)

**What it does:**
- 4 strategies: DipArb, Copy Trading, Smart Money, Market Making
- 4-layer risk protection, dynamic position sizing
- Whale detection, gas fee accounting
- v3.1 (Jan 2026) with enhanced risk management

**Implementability:** 3/5 — Easy setup but Node.js, strategy quality uncertain.

---

### 7. PolyBot Reverse-Engineering Toolkit
**Source:** https://github.com/ent0n29/polybot
**Recommendation:** MEDIUM (queued)

**What it does:**
- Java 21 microservices for strategy reverse-engineering
- ClickHouse + Redpanda event pipeline
- AWARE product layer for trader intelligence
- Full monitoring stack (Grafana, Prometheus)

**Implementability:** 2/5 — Heavy infra, overkill for most use cases.

---

### 8. Rust ML + LLM Trading Bot
**Source:** https://github.com/skharchikov/polymarket-bot
**Recommendation:** MEDIUM (queued)

**What it does:**
- Two Rust bots: ML-driven (XGBoost + LLM consensus + Bayesian) and copy-trading
- 29 market features, Kelly sizing, historical backtesting crawler
- Telegram notifications, Grafana monitoring

**Implementability:** 2/5 — Research-grade. Complex Rust+Python setup.
