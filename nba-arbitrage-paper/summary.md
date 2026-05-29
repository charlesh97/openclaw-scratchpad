# Arbitrage Analysis in Polymarket NBA Markets

**Authors:** Guang Cheng, Jiaxin Yang et al.
**Source:** https://arxiv.org/abs/2605.00864
**Published:** April 22, 2026
**Recommendation:** YES (empirical arb analysis)

## Summary

Systematic empirical analysis of algorithmic arbitrage within Polymarket's NBA game markets. Reconstructs continuous market states from **75+ million limit order book snapshots across 173 games**.

### Key Findings:

- **Single-market anomalies**: Extremely rare — only 7 executable in-game episodes, median duration of just **3.6 seconds**
- **Combinatorial inefficiencies**: More frequent — 290 active episodes, concentrated in final minutes of live play
- **Combinatorial returns**: Median return of **101 basis points** — statistically meaningful
- **The "Middle" jackpot**: Never empirically realized (theoretical YES and NO combined arb)
- **Severe liquidity bottleneck**: 76.9% of combinatorial opportunities constrained to average executable size of just **14.8 shares**

### Bottom Line

"While executable mispricings exist, they are structurally bounded by liquidity, confining risk-free extraction strictly to the retail scale."

### Relevance

Provides realistic expectations for arb profitability on Polymarket. The liquidity constraint (14.8 shares average) means manual retail-scale arbitrage is possible, but institutional-scale extraction is not. Confirms the limits-to-arbitrage framework applies to DeFi prediction markets.

### Key Takeaway

Arbitrage is real but small — expect retail-scale profits only. The 101bps median return on combinatorial arb is good but limited by depth.
