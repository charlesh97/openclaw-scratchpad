## 2026-05-13 Research Findings

### 1. Polymarket AI Agent Framework
**Source:** https://github.com/Polymarket/agents
**Recommendation:** YES

**What it does:**
- Official Polymarket SDK for building autonomous AI trading agents
- Connects LLMs to Polymarket's CLOB and Data APIs
- Tool-based architecture: news retrieval, data querying, trade execution
- Extensible plugin system for custom strategies

**Implementability:** 5/5

---

### 2. Prediction Market Maker — Paradigm Challenge (#2)
**Source:** https://github.com/octavi42/prediction-market-maker
**Recommendation:** YES

**What it does:**
- Battle-tested market-making strategy that placed #2 in Paradigm's competition
- 110 iterations over 8 hours of live competition
- Covers quoting mechanics, adverse selection, inventory risk, order sizing
- Complete case study in prediction market microstructure

**Implementability:** 4/5

---

### 3. Cross-Platform Arbitrage Bot (Polymarket ↔ Kalshi)
**Source:** https://github.com/realfishsam/prediction-market-arbitrage-bot
**Recommendation:** MEDIUM

**What it does:**
- Detects and executes arb between Polymarket and Kalshi
- Uses pmxt.dev as unified API bridge
- Fee-aware profit calculations
- Real-time monitoring for price discrepancies

**Implementability:** 4/5

---

### 4. BTC 15-Minute Trading Bot
**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot
**Recommendation:** MEDIUM

**What it does:**
- Production-grade 7-phase architecture for Polymarket BTC 15-min markets
- Multi-signal approach: technical, on-chain, sentiment
- Kelly criterion position sizing, stop-loss, self-learning
- Smart order routing to minimize slippage

**Implementability:** 4/5

---

### 5. Copy Trading Bot (Rust)
**Source:** https://github.com/gamma-trade-lab/polymarket-copy-trading-bot
**Recommendation:** MEDIUM

**What it does:**
- Rust-based copy trading bot mirroring top Polymarket traders
- Low-latency detection using dedicated Polygon RPC
- Benefits from early 2026 removal of 500ms taker order delay
- Per-trader limits and drawdown protection

**Implementability:** 3/5

---

### 6. Polybot — Strategy Reverse-Engineering Toolkit
**Source:** https://github.com/ent0n29/polybot
**Recommendation:** MEDIUM

**What it does:**
- Open-source infrastructure for reverse-engineering Polymarket strategies
- Market data foundation + AWARE intelligence layer
- Trader analytics, PSI indices, fund mirroring
- API-first design for integration

**Implementability:** 3/5

---

### 7. Polymarket-Kalshi BTC Arbitrage Bot
**Source:** https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot
**Recommendation:** MEDIUM

**What it does:**
- Focused BTC 1-Hour cross-platform arb between Polymarket and Kalshi
- Combines opposing positions for guaranteed profit when combined cost < $1.00
- CLOB + REST API integration
- Hourly resolution provides frequent, predictable opportunities

**Implementability:** 4/5

---

### 8. SoK: Market Microstructure for DePMs
**Source:** https://arxiv.org/abs/2510.15612
**Recommendation:** MEDIUM

**What it does:**
- Comprehensive academic survey of DePM microstructure
- Taxonomy of market types, pricing mechanisms, arb strategies
- Cross-venue comparison framework
- Essential theoretical foundation for bot strategy design

**Implementability:** 2/5

---

### 9. Unravelling the Probabilistic Forest: Arbitrage on Polymarket
**Source:** https://arxiv.org/abs/2508.03474
**Recommendation:** YES

**What it does:**
- First large-scale empirical study of Polymarket arb
- Quantifies $40M USD of arbitrage profit extracted
- Identifies two arb types: Market Rebalancing (intra-market) and Combinatorial (inter-market)
- 62% of LLM-detected dependencies fail to generate profit

**Implementability:** 3/5

---

### 10. Anatomy of a Decentralized Prediction Market
**Source:** https://arxiv.org/abs/2604.24366
**Recommendation:** MEDIUM

**What it does:**
- Tick-level microstructure analysis: 30B order book events, 600 markets
- Eight stylized facts including longshot premium, wash trade patterns
- Confirms arb exists but is structurally bounded by liquidity
- Critical empirical baseline for bot design

**Implementability:** 2/5

---

### 11. Fill-Side Non-Retail Trading on Polymarket
**Source:** https://arxiv.org/abs/2605.11640
**Recommendation:** MEDIUM

**What it does:**
- Most current study (May 2026) of professional trading on Polymarket
- Behavioral tiers of market makers and arbitrageurs
- Quote-attribution methodology for strategy detection
- Competitive intelligence for positioning new bots

**Implementability:** 2/5
