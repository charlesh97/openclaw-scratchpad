## 2026-05-31 Research Findings

### 1. Polymarket ↔ Kalshi Cross-Platform Arbitrage Bot
**Source:** https://github.com/realfishsam/prediction-market-arbitrage-bot
**Recommendation:** YES

**What it does:**
- Detects price differences between Polymarket and Kalshi for identical events
- Automatically buys low on one platform and sells high on the other
- Built on pmxt.dev — a cross-platform prediction market exchange toolkit
- Risk-free arbitrage since events resolve identically across platforms

**Implementability:** 4/5
Python, well-documented codebase. Directly applicable to our arb bot project.

---

### 2. Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets
**Source:** https://arxiv.org/abs/2508.03474
**Recommendation:** YES

**What it does:**
- First large-scale empirical study of arbitrage on Polymarket
- Identifies two arb types: Market Rebalancing (YES+NO < $1) and Combinatorial (across logically linked markets)
- Uses on-chain order book data to quantify real opportunity sizes
- Provides mathematical framework for systematic arb detection

**Implementability:** 3/5
Theoretical paper — no code but provides the detection framework. Combinatorial arb is complex to implement but the simpler Market Rebalancing arb is trivially implementable.

---

### 3. Polymarket BTC 15-Minute Trading Bot (7-Phase Architecture)
**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot
**Recommendation:** MEDIUM

**What it does:**
- Production-grade algorithmic trading for BTC 15-min prediction markets
- 7-phase architecture: data ingestion, signal generation, probability estimation, position sizing, execution, monitoring, self-learning
- Combines technical, on-chain, and sentiment signals

**Implementability:** 4/5
Python + standard ML. Excellent architecture template.

---

### 4. Polymarket Trading Bot (5 Strategies: Hedging, Micro-spreads, LP, Arb, Low-volume)
**Source:** https://github.com/Anmoldureha/polymarket-trading-bot-strategies
**Recommendation:** MEDIUM

**What it does:**
- 5 sophisticated trading strategies in one codebase
- Includes risk management, paper trading, Telegram alerts
- Multi-strategy approach (arb + LP + hedging)

**Implementability:** 3/5
TypeScript (not Python). Strategies are well-documented but need porting.

---

### 5. Polybot — Polymarket Strategy Reverse-Engineering Toolkit
**Source:** https://github.com/ent0n29/polybot
**Recommendation:** MEDIUM (queue)

**What it does:**
- Open-source trading infrastructure and strategy reverse-engineering
- Fund mirroring, PSI sentiment index, top-trader analysis
- Low-latency execution layer

**Implementability:** 3/5
Python + Web3. More research tool than turnkey bot.

---

### 6. The Anatomy of a Decentralized Prediction Market
**Source:** https://arxiv.org/abs/2604.24366
**Recommendation:** MEDIUM (queue)

**What it does:**
- Microstructure study of Polymarket using 30B order book events
- 8 stylized facts about liquidity, spreads, wash trading
- Directly informs bot parameter tuning (minimum spreads, category-specific settings)

**Implementability:** 3/5
Research paper — empirical findings, not code. Invaluable for parameter tuning.

---

### 7. OctoBot Prediction Market Module
**Source:** https://github.com/Drakkar-Software/OctoBot-Prediction-Market
**Recommendation:** MEDIUM (queue)

**What it does:**
- Polymarket module for the OctoBot trading framework
- Copy trading + arbitrage strategies
- Full web dashboard and backtesting

**Implementability:** 5/5
Python, battle-tested framework, immediately deployable.

---

### 8. SoK: Market Microstructure for DePMs
**Source:** https://arxiv.org/abs/2510.15612
**Recommendation:** NO (background reference)

**What it does:**
- Comprehensive survey of prediction market microstructure
- Theoretical foundation for empirical work

**Implementability:** 2/5
Pure survey. Background reading.

---

### 9. Paradigm Challenge #2 — Prediction Market Maker
**Source:** https://github.com/octavi42/prediction-market-maker
**Recommendation:** MEDIUM (queue)

**What it does:**
- Market making strategy for prediction markets
- 110 iterations over 8 hours of competition
- Covers quoting, adverse selection, inventory risk

**Implementability:** 4/5
Well-documented Python. Market making complements arb strategy.

---

### 10. Awesome Prediction Market Tools
**Source:** https://github.com/aarora4/Awesome-Prediction-Market-Tools
**Recommendation:** NO (reference)

**What it does:**
- Curated directory of prediction market tools
- Lists Eventarb, Polytrage, PolyInsider and other arb-relevant tools

**Implementability:** N/A
Reference directory. But Eventarb is a direct competitor to study.
