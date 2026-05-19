## 2026-05-19 Research Findings

### 1. Combinatorial Arbitrage Framework
**Source:** [arXiv:2508.03474](https://arxiv.org/abs/2508.03474)
**Recommendation:** YES

**What it does:**
- First large-scale empirical analysis of arbitrage on Polymarket (86M transactions, 17,218 conditions)
- Identifies Market Rebalancing Arbitrage (YES+NO ≠ $1.00) found in 41% of all markets
- Identifies Combinatorial Arbitrage spanning multiple related conditions
- ~$40M in extracted profits documented between Apr 2024-Apr 2025

**Implementability:** 4/5

---

### 2. Cross-Platform Arbitrage Bot (Polymarket + Kalshi)
**Source:** [ImMike/polymarket-arbitrage](https://github.com/ImMike/polymarket-arbitrage) + [realfishsam/prediction-market-arbitrage-bot](https://github.com/realfishsam/prediction-market-arbitrage-bot) + [CarlosIbCu](https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot)
**Recommendation:** YES

**What it does:**
- Detects price differences between Polymarket and Kalshi for the same prediction market event
- Executes synthetic arbitrage: buy YES cheap + buy NO cheap, lock in profit
- Multiple reference implementations in Python, Node.js, Rust with live dashboards
- Handles 5,000+ market scanning, fuzzy matching, fee accounting

**Implementability:** 4/5

---

### 3. BTC 15-Minute Signal Fusion Trading Bot
**Source:** [aulekator/Polymarket-BTC-15-Minute-Trading-Bot](https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot)
**Recommendation:** MEDIUM

**What it does:**
- Production-grade 7-phase architecture with multi-signal intelligence (spike, sentiment, divergence)
- Self-learning weight optimization, Grafana monitoring, auto-recovery
- Conservative risk management ($1 max/trade, 30% stop loss)

**Implementability:** 3/5

---

### 4. Paradigm Market Maker (octavi42)
**Source:** [octavi42/prediction-market-maker](https://github.com/octavi42/prediction-market-maker)
**Recommendation:** MEDIUM

**What it does:**
- #2 place in Paradigm's Prediction Market Challenge (Apr 2026)
- Market-making strategy: monopoly regime (~60% of edge), volatility-adjusted quoting, inventory management
- Comprehensive case study on prediction market microstructure

**Implementability:** 4/5

---

### 5. PredictOS Framework
**Source:** [PredictionXBT/PredictOS](https://github.com/PredictionXBT/PredictOS)
**Recommendation:** MEDIUM

**What it does:**
- Open-source all-in-one framework for prediction market AI agents
- Supports Polymarket, Kalshi, Jupiter; self-hosted, private strategies
- Auto-arb detection by pasting market URLs

**Implementability:** 3/5

---

### 6. Polymarket 4-Strategy Bot
**Source:** [MrFadiAi/Polymarket-bot](https://github.com/MrFadiAi/Polymarket-bot)
**Recommendation:** MEDIUM

**What it does:**
- 4 strategies (Copy Trading, Dip Arb, Smart Money, Auto-Rotation)
- 4-layer risk protection system
- Active development (v3.1 Jan 2026)

**Implementability:** 3/5

---

### 7. Rust Cross-Market Arbitrage Bot
**Source:** [Trum3it/polymarket-arbitrage-bot](https://github.com/Trum3it/polymarket-arbitrage-bot)
**Recommendation:** NO (queue)

**What it does:**
- Rust-based arb for ETH/BTC 15-minute markets
- Market-neutral: buy Up on one + Down on the other

**Implementability:** 2/5

---

### 8. Polybot Java Infrastructure
**Source:** [ent0n29/polybot](https://github.com/ent0n29/polybot)
**Recommendation:** NO (queue)

**What it does:**
- Java 21 microservices with ClickHouse, Redpanda for strategy reverse-engineering
- Over-engineered for our current needs

**Implementability:** 2/5

---

### Research Papers

| # | Paper | Link | Relevance |
|---|-------|------|-----------|
| 1 | Unravelling the Probabilistic Forest | [arXiv:2508.03474](https://arxiv.org/abs/2508.03474) | **Core** — $40M arb analysis |
| 2 | Anatomy of a DePM | [arXiv:2604.24366](https://arxiv.org/abs/2604.24366) | **Core** — 30B event microstructure study |
| 3 | Toward Black Scholes for Prediction Markets | [arXiv:2510.15205](https://arxiv.org/abs/2510.15205) | **Supporting** — pricing theory |
| 4 | SoK: Market Microstructure for DePMs | [arXiv:2510.15612](https://arxiv.org/abs/2510.15612) | **Supporting** — survey |
| 5 | Arbitrage Analysis in NBA Markets | [arXiv:2605.00864](https://arxiv.org/abs/2605.00864) | **Supporting** — sports arb liquidity |
