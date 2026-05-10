# SoK: Market Microstructure for Decentralized Prediction Markets (DePMs)

**Source:** https://arxiv.org/abs/2510.15612
**Type:** Systematization of Knowledge paper
**Recommendation:** MEDIUM

## What It Does

Survey paper systematizing knowledge about market microstructure in decentralized prediction markets (DePMs). Key contributions:

1. **Framework definition** — Defines what makes DePM microstructure different from centralized prediction markets
2. **Cross-venue mapping** — Frames open questions across venues without matching tick-level data
3. **Sub-types taxonomy** — Categorizes different prediction market mechanisms (merging/splitting, automated bookmaking)
4. **Deregulation insight** — "Deregulatory trajectory of 2025–2026 may improve liquidity while systematically degrading the epistemic quality of the public signals"

## Why It Matters

Important context for understanding the structural differences between DePMs (like Polymarket) and traditional markets. Key insight: 
- Regulatory changes 2025–2026 may increase trading volume and liquidity
- But the quality of price signals as information aggregation mechanisms may decline
- Has implications for which markets are "efficient" vs. exploitable

## Implementability: 2/5

Academic survey/framework paper. Not actionable for building a trading system directly, but useful for understanding the structural landscape and which market properties to analyze.

**Next steps:**
1. Skim the paper to understand the DePM microstructure taxonomy
2. Use the framework to identify which market types on Polymarket are most likely to have exploitable inefficiencies
3. Monitor the regulatory trajectory for opportunities as new market types open up