## 2026-06-30 Kalshi-Focused Sweep

### Tier 1 — Must Read

#### A. "Makers and Takers: The Economics of the Kalshi Prediction Market"
**Source:** https://www.karlwhelan.com/Papers/Kalshi.pdf
**Authors:** Bürgi, Deng, Whelan — UCD, January 2026
**Recommendation:** YES ✅ (definitive Kalshi reference)

First systematic evidence on Kalshi pricing. Transaction-level data on 300,000+ contracts.
- Clear favorite-longshot bias
- Makers earn higher returns than Takers
- Quote-driven microstructure model
- Kalshi-specific: every trade is Maker-initiated offer + Taker acceptance
- Modest disagreement + small probability overstatement reproduces patterns

**Implementability:** N/A (academic reference)

---

#### B. "Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics"
**Source:** https://arxiv.org/abs/2602.19520
**Author:** Nam Anh Le, February 2026
**Recommendation:** YES ✅ (Kalshi calibration bible)

292M trades, 327K binary contracts across Kalshi + Polymarket.
- 4 components explain 87.3% of calibration variance
- Trade-size scale effect: large trades amplify underconfidence in politics on Kalshi (Δ=0.53) but NOT Polymarket (Δ=0.11)
- Critical for understanding when Kalshi prices are systematically miscalibrated

**Implementability:** N/A (academic reference)

---

### Tier 2 — High Value

#### C. "When Do Markets Fully Process Public Information?"
**Source:** https://arxiv.org/abs/2606.07811
**Date:** June 2026
**Recommendation:** YES

Real-time Kalshi market efficiency study.
- Prices adjust only 0.64-for-one relative to benchmark
- Underreaction predicts drift
- Liquidity + salience interaction drives incomplete adjustment

**Implementability:** N/A (academic reference)

---

#### D. "NegRisk Market Rebalancing: How $29M Was Extracted"
**Source:** https://navnoorbawa.substack.com/p/negrisk-market-rebalancing-how-29m
**Author:** Navnoor Bawa, November 2025
**Recommendation:** YES

Multi-condition arb on Kalshi NegRisk markets.
- 29× capital efficiency vs binary arb
- Buy NO dominance (60% of profits)
- Top arbitrageur: $2.01M across 4,049 txns ($496 avg)

**Implementability:** 4/5

---

#### E. Kalshi Official API
**Source:** https://docs.kalshi.com/
- REST API + FIX protocol
- Market data, order placement, trade history
- Rate limits apply
- Dev guide: https://dev.to/zuplo/kalshi-api-the-complete-developers-guide-1fo4

**Implementability:** 5/5

---

### Tier 3 — Ecosystem & Practical

#### F. Kalshi Fee Structure (June 2026)
**Source:** https://kalshi.com/docs/kalshi-fee-schedule.pdf | https://kalshifees.com/
- Per-contract fee: 1-7¢ depending on price
- Highest near 50/50, lowest near extremes
- Fees WAIVED on losing trades

#### G. Open-Source Kalshi Bot Ecosystem (30+ Projects)
**Source:** https://www.botforkalshi.com/blog/open-source-kalshi-bot-ecosystem
Published May 2026. Comprehensive catalog.

#### H. 7 Proven Kalshi Trading Strategies
**Source:** https://www.botforkalshi.com/blog/kalshi-trading-strategies-guide
Weather model divergence, sports prop value hunting, news-fade, cross-platform arb.

#### I. New GitHub Repos (not in prior sweep)
- rao/pm-kalshi-arbitrage-bot: https://github.com/rao/pm-kalshi-arbitrage-bot
- antmlap/kalshi-arbitrage-bot: https://github.com/antmlap/kalshi-arbitrage-bot
- TopTrenDev/polymarket-kalshi-arbitrage-bot: https://github.com/TopTrenDev/polymarket-kalshi-arbitrage-bot

#### J. Institutional & Data
- Jump Trading market making on Kalshi
- Jon-Becker prediction-market-analysis: https://github.com/Jon-Becker/prediction-market-analysis
- PredictEngine guide: https://www.predictengine.ai/blog/algorithmic-kalshi-trading-in-2026-the-complete-guide
- DFlow Kalshi on Solana: https://www.quicknode.com/guides/solana-development/3rd-party-integrations/kalshi-prediction-markets-with-dflow
