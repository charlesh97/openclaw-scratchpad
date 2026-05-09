# polymarket-arbitrage (ImMike) — Cross-Platform + Bundle Arb

**Source:** https://github.com/ImMike/polymarket-arbitrage
**Recommendation:** MEDIUM (strong code, lower priority than #1)

## What it does

Python-based bot that runs **three arbitrage modes**:

1. **Cross-Platform (Polymarket ↔ Kalshi):** Detects price differences on the same event across both platforms — same mechanism as `realfishsam/prediction-market-arbitrage-bot` but with bundle detection added.
2. **Bundle Arbitrage (YES + NO ≠ $1.00):** Monitors individual Polymarket markets where YES + NO token prices don't sum to ~$1.00. When the bundle diverges, buys the undervalued combination and sells the overvalued one.
3. **Market Making:** Places competitive bid/ask orders to capture spreads, using a simple inventory-based quoting strategy.

## Why it matters

- The combination of all three modes makes this a more complete arbitrage system than single-strategy bots
- "Bundle arbitrage" is the purest form of prediction market arb — exploiting the AMM's own internal pricing failure
- Market-making mode provides a second revenue stream even when pure arb opportunities are thin
- Python is accessible and the repo is self-contained

## Risks

- Market-making mode carries inventory risk — if prices move adversely, the bot holds a losing position
- Bundle arbitrage opportunities are fleeting; the paper found 76.9% of episodes have max executable size of ~14.8 shares — very small
- No mention of transaction cost modeling (gas, fees, slippage) which can easily exceed the arbitrage profit on small positions

## Implementability: 3/5

- Full Python implementation available — high transparency
- Requires Polymarket API + optional Kalshi API keys
- The market-making component requires more careful risk management than the arb component
- No recent commit date visible from search — need to check for activity

## Next Steps

1. Clone and audit the full Python codebase for completeness
2. Add gas/slippage cost modeling to the bundle arb profitability calculator
3. Backtest bundle arb profitability across Q4 2025 — Q1 2026 data
4. Add position sizing with Kelly criterion for market-making legs

---
*Part of vega's arb-bot-analysis research.*