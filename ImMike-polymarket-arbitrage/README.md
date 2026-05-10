# ImMike/polymarket-arbitrage

**Source:** https://github.com/ImMike/polymarket-arbitrage
**Type:** Production arbitrage bot
**Recommendation:** YES — Top of Email

## What It Does

The most comprehensive open-source Polymarket arbitrage system found today:

1. **Cross-Platform Arbitrage** — Detects price differences between Polymarket and Kalshi for the same prediction event. Buy low on one platform, sell high on the other.

2. **Bundle Arbitrage Detection** — Identifies when YES + NO shares don't sum to ~$1.00 (within a single market). Classic LP arb inside Polymarket's own order book.

3. **Market Making** — Captures bid-ask spreads by placing competitive limit orders on both sides of the book.

4. **Scale** — Monitors 10,000+ markets simultaneously, calculating arb across thousands of combinations per second.

5. **Speed** — Sub-50ms order placement when opportunity detected.

## Why It Matters

This is the highest-scope, most production-ready arb bot found in today's research. The combination of:
- 10k+ market simultaneous monitoring
- Both cross-platform (Polymarket↔Kalshi) and intra-market (bundle) arb modes
- Market-making component for spread capture
- Sub-50ms execution

...makes this the most actionable system to study and potentially adapt.

## Architecture Reference

- Monitors Polymarket + Kalshi APIs simultaneously
- Price discrepancy detection engine
- Bundle sum detection (YES + NO ≠ 1.0 within tolerance)
- Order execution layer (likely CEX + Polymarket API)
- Position tracking and P&L

## Implementability: 4/5

Strong production reference. Clone and audit the code for:
1. The market scanning loop (how does it handle 10k markets efficiently?)
2. The opportunity detection thresholds (what spread triggers execution?)
3. The execution path (order placement, confirmation handling, slippage)
4. The cross-platform funding flow (how does it move capital between Polymarket and Kalshi?)

**Next steps:**
1. Clone and run the repo
2. Read the main bot logic to understand the detection/execution pipeline
3. Identify which parts are Polymarket-specific vs. generalizable
4. Assess Kalshi API integration complexity