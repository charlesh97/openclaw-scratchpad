# Combinatorial Arbitrage in Polymarket NBA Markets

**Source:** https://arxiv.org/html/2605.00864
**Published:** ~May 2026
**Recommendation:** YES (email candidate #1 — quantitative rigor)

## What it does

This arXiv paper presents the **first systematic empirical analysis of combinatorial arbitrage** in live Polymarket NBA markets. The authors analyze:

- **Moneyline–Spread pairs:** NBA games have both "who wins" (moneyline) and "does the spread cover" (spread) markets. These are not independent — they share probabilistic structure. When prices don't respect this structure, arbitrage exists.
- **Market Rebalancing Arb:** Within a single market, when YES + NO ≠ $1.00
- **Combinatorial Arb:** Cross-market, when logically related contracts have inconsistent prices

They find ~290 active combinatorial arbitrage episodes concentrated in the final minutes of live play — rapid scoring events cause cross-market dislocations.

## Key Quantitative Findings

- **76.9%** of combinatorial episodes had a max executable size of **~14.8 shares** — severely liquidity-constrained
- Limits-to-arbitrage (Shleifer & Vishny, 1997) confirmed empirically: shallow order book depth is THE binding constraint
- Combinatorial arb is more frequent than market rebalancing arb but even smaller in executable size
- The "62% of LLM-detected dependencies fail to generate profit" (Navnoor Bawa, Medium 2025) finding is confirmed: most theoretical arbs aren't executable at scale

## Why it matters for our strategy

- Provides the **most accurate real-world constraints** on what a Polymarket arbitrage bot can actually achieve
- The finding that liquidity is the binding constraint means our bot needs to focus on **high-frequency, small-size execution** rather than large capital deployment
- The final-minutes-of-live-play concentration suggests that **in-play NBA markets** are the highest-opportunity windows
- This is peer-reviewed empirical evidence — should drive our position sizing assumptions directly

## Paper URL
https://arxiv.org/html/2605.00864

## Implementability: N/A (research paper — not a bot)

The paper itself is not a bot, but its findings directly inform our strategy:
- Position sizing must be calibrated to ~14.8 share executable capacity
- Focus on NBA live markets during final minutes, when dislocations are most frequent
- Do not expect large-scale risk-free arb — the market is efficiently constrained by liquidity

## Next Steps

1. Download and read full paper
2. Extract the specific moneyline-spread combinatorial relationship formulas
3. Build an NBA-specific arb scanner that monitors live NBA Polymarket markets in final minutes
4. Calibrate Kelly criterion sizing to the 14.8 share liquidity ceiling

---
*Part of vega's arb-bot-analysis research. Found via combinatorial arbitrage search.*