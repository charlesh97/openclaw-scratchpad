## 2026-05-12 Research Findings

### 1. OctoBot Prediction Market
**Source:** https://github.com/Drakkar-Software/OctoBot-Prediction-Market
**Recommendation:** YES

**What it does:**
- Open-source Polymarket trading bot built on the OctoBot framework (established crypto trading bot platform)
- Copy trading — mirrors top Polymarket leaderboard profiles automatically
- Bundle arbitrage detection (YES+NO < $1.00) — currently under development
- Paper trading mode for strategy testing before deploying real funds
- Full visual UI + Telegram interface + upcoming mobile app
- Self-custody: private keys stay on device
- Kalshi integration in development for cross-platform arbitrage

**Implementability:** 4/5

---

### 2. PIRAP: Resolution-Aware Perpetual Futures Framework
**Source:** https://arxiv.org/abs/2605.10400 (May 11, 2026 — published yesterday)
**Recommendation:** YES

**What it does:**
- First formal framework for perpetual futures on binary prediction market probabilities
- Six components: index estimator, jump-aware margin, leverage compression, resolution-aware funding, halt protocol, eligibility gates
- Full empirical evaluation using Polymarket PMXT v2 archive (13,298 markets, April 21-27, 2026)
- Code available: github.com/ForesightFlow/event-linked-perps
- Identifies fundamental structural limitations: standard perpetual frameworks fail on bounded-event underlyings
- Explicitly non-deployable in current form, but provides blueprint for future work

**Implementability:** 2/5 (research-grade, but structural insights are valuable)

---

### 3. Polymarket Arbitrage Bot (ImMike)
**Source:** https://github.com/ImMike/polymarket-arbitrage
**Recommendation:** MEDIUM (queue)

**What it does:**
- Python bot scanning 10,000+ Polymarket markets in real-time
- Bundle arbitrage detection (YES+NO < $1.00)
- Cross-platform arbitrage (Polymarket ↔ Kalshi) using text similarity matching
- Market making via competitive bid/ask orders
- Live FastAPI web dashboard + simulation mode
- Risk management with position limits, loss limits, kill switch

**Implementability:** 4/5

---

### 4. Polymarket BTC 15-Minute Trading Bot
**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot
**Recommendation:** MEDIUM (queue)

**What it does:**
- Production-grade bot for Polymarket's short-term BTC markets
- 7-phase architecture: multi-exchange data ingestion, spike/sentiment/divergence signal processors, fusion engine
- Grafana dashboards + Prometheus metrics + Redis mode switching
- Self-learning signal weight optimization based on performance

**Implementability:** 3/5

---

### 5. Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets
**Source:** https://arxiv.org/abs/2508.03474
**Recommendation:** MEDIUM (queue)

**What it does:**
- Large-scale analysis of arbitrage on Polymarket
- Identifies Market Rebalancing Arbitrage (intra-market) and Combinatorial Arbitrage (inter-market)
- Found $40M in realized arbitrage profits extracted between April 2024 and April 2025
- Uses on-chain historical order book data

**Implementability:** 3/5

---

### 6. The Anatomy of a Decentralized Prediction Market
**Source:** https://arxiv.org/abs/2604.24366
**Recommendation:** MEDIUM (queue)

**What it does:**
- First tick-level microstructure study using Polymarket's WebSocket feed (30B events over 52 days)
- Eight stylized facts: longshot spread premium, depth-concentration profile, null block-clock alignment, broad maker diversity
- Median 1% self-counterparty wash share, 22% upper tail
- Sub-50ms median ingestion delay

**Implementability:** 2/5 (academic study, but valuable microstructure insights for bot design)

---

### 7. PredictOS — All-in-One Prediction Market Framework
**Source:** https://github.com/PredictionXBT/PredictOS
**Recommendation:** MEDIUM (queue)

**What it does:**
- Open-source framework for prediction markets
- Cross-platform arbitrage: paste any market URL, auto-searches for same market on other platforms
- Real-time monitoring and profit calculations
- Modular architecture for extensibility

**Implementability:** 3/5

---

### 8. Polymarket AI Agents Framework
**Source:** https://github.com/Polymarket/agents
**Recommendation:** MEDIUM (queue)

**What it does:**
- Official Polymarket framework for AI agent trading
- 45+ tools for market analysis, news retrieval, and execution
- MCP server integration for LLM-powered trading
- Enterprise-grade safety features

**Implementability:** 3/5

---

### 9. Prediction Market Maker (Paradigm Challenge)
**Source:** https://github.com/octavi42/prediction-market-maker
**Recommendation:** MEDIUM (queue)

**What it does:**
- Market making strategy that placed #2 in Paradigm's Prediction Market Challenge
- 110 iterations, 8 hours
- Detailed case study of market making mechanics: quoting, adverse selection, inventory risk, order sizing

**Implementability:** 3/5

---

### 10. Arbitrage Analysis in Polymarket NBA Markets
**Source:** https://arxiv.org/abs/2605.00864
**Recommendation:** MEDIUM (queue)

**What it does:**
- Studies combinatorial arbitrage across Moneyline–Spread pairs in live sports
- 290 active arbitrage episodes, concentrated in final minutes of play
- 76.9% of opportunities constrained to average 14.8 shares executable
- Demonstrates limits-to-arbitrage friction in decentralized prediction markets

**Implementability:** 2/5 (academic, but liquidity constraint data is actionable)
