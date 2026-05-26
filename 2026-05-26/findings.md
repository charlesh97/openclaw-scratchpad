## 2026-05-26 Research Findings

### 1. Polymarket BTC 15-Minute Trading Bot
**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot
**Recommendation:** YES

**What it does:**
- Production-grade algorithmic bot for Polymarket's 15-minute BTC prediction markets
- 7-phase architecture: Ingestion → Nautilus Core → Signal Processors → Fusion Engine → Risk → Execution → Monitoring/Learning
- Multi-signal intelligence: Spike Detection, Sentiment Analysis, Price Divergence
- Self-learning weight optimization — automatically tunes signal weights based on performance
- Grafana dashboards, Prometheus metrics, auto-recovery

**Implementability:** 4/5

---

### 2. Polymarket-Kalshi Arbitrage Bot
**Source:** https://github.com/ImMike/polymarket-arbitrage
**Recommendation:** YES

**What it does:**
- Cross-platform arbitrage bot scanning 5,000+ Polymarket and Kalshi markets
- Three strategies: Cross-Platform Arbitrage, Bundle Arbitrage, Market Making
- 99.6% win rate in simulation, $573 simulated profit
- Live dashboard, fee accounting, dual data modes (real/simulation)

**Implementability:** 4/5

---

### 3. Unravelling the Probabilistic Forest (ArXiv)
**Source:** https://arxiv.org/abs/2508.03474
**Recommendation:** YES

**What it does:**
- First large-scale analysis of arbitrage on Polymarket (86M transactions, 17,218 conditions)
- Identifies Market Rebalancing Arb and Combinatorial Arb
- ~$40M in extracted arbitrage profits over study period
- Integer programming framework for combinatorial detection

**Implementability:** 4/5

---

### 4. Polymarket Trading Bot Strategies (PolyHFT)
**Source:** https://github.com/Anmoldureha/polymarket-trading-bot-strategies
**Recommendation:** MEDIUM

**What it does:**
- 10 distinct trading strategies: hedging, micro-spreads, LP, arb, low-volume, spread scalping, tail-end, combinatorial arb, legged arb, continuous MM
- Enterprise risk management with multi-layered controls
- Paper trading, Telegram notifications, parallel market fetching

**Implementability:** 3/5

---

### 5. Prediction Market Maker (Paradigm #2)
**Source:** https://github.com/octavi42/prediction-market-maker
**Recommendation:** MEDIUM

**What it does:**
- #2 in Paradigm's Prediction Market Challenge (score: $41.09 mean edge)
- 110 strategy iterations in 8 hours
- Key insight: Monopoly regime (~60% of edge), volatility-adjusted quoting, inventory skew
- Complete case study in prediction market microstructure

**Implementability:** 3/5

---

### 6. Polymarket-Kalshi BTC Arbitrage Bot
**Source:** https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot
**Recommendation:** MEDIUM

**What it does:**
- Focused BTC 1-Hour cross-platform arb scanner
- FastAPI backend + Next.js dashboard
- Includes detailed arbitrage thesis

**Implementability:** 4/5

---

### 7. Polybot — Strategy Reverse-Engineering Toolkit
**Source:** https://github.com/ent0n29/polybot
**Recommendation:** NO

**What it does:**
- Java microservices for Polymarket data ingestion, strategy execution, and analytics
- ClickHouse + Redpanda event pipeline
- Strategy reverse-engineering and replication scoring
- Foundation for AWARE trader intelligence product

**Implementability:** 2/5

---

### 8. Anatomy of a Decentralized Prediction Market (ArXiv)
**Source:** https://arxiv.org/abs/2604.24366
**Recommendation:** MEDIUM

**What it does:**
- 30B order book events across 600 markets over 52 days
- Eight stylized facts about Polymarket microstructure
- Category-specific spread differences, wash trading analysis

**Implementability:** 3/5

---

### 9. Toward Black-Scholes for Prediction Markets (ArXiv)
**Source:** https://arxiv.org/abs/2510.15205
**Recommendation:** NO

**What it does:**
- Proposes logit jump-diffusion pricing model for prediction markets
- Risk-neutral martingale framework for belief volatility and jump intensity
- Theoretical foundation for prediction market derivatives

**Implementability:** 1/5
