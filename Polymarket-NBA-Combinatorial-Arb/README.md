# Polymarket NBA Combinatorial Arbitrage

**Source:** https://arxiv.org/html/2605.00864
**Type:** Empirical research paper
**Recommendation:** MEDIUM

## What It Does

Empirical study analyzing combinatorial arbitrage opportunities in Polymarket NBA markets (Moneyline–Spread pairs). Key findings:

1. **290 active arbitrage episodes identified** — concentrated in final minutes of live games
2. **Trigger mechanism** — Abrupt scoring events cause rapid cross-market dislocations
3. **Size ceiling** — 76.9% of episodes constrained to avg executable size of just **14.8 shares**
4. **Limits-to-arbitrage confirmed** — Shleifer-Vishny friction: shallow order book depth prevents full correction even when mispricings are unambiguously identified

## Why It Matters

This paper provides the **most important structural constraint** for any Polymarket arbitrage system: **liquidity is the binding ceiling**. Even when you detect a perfect arb opportunity:
- Order book depth limits how much you can actually trade
- Larger positions move the price against you mid-execution
- Net P&L after slippage may be negative

This is confirmed in sports markets where contracts resolve within hours (less time for liquidity to accumulate) and in political markets (Tsang and Yang 2026).

## Key Quantitative Findings

| Metric | Value |
|--------|-------|
| Active episodes (NBA Moneyline-Spread) | 290 |
| % constrained to small size | 76.9% |
| Avg executable size | ~14.8 shares |
| Timing | Concentrated in final minutes of live play |

## Implementability: 3/5

Not a deployable strategy — it's an empirical ceiling study. Read it to calibrate your position sizing expectations. Don't expect to find $50k arb opportunities; the liquidity just isn't there for most of the market.

**Next steps:**
1. Read the paper to understand the methodology
2. Model your position sizing to respect the ~15 share average ceiling
3. Focus arb efforts on higher-liquidity markets (US elections, major economic data) where depth is deeper
4. Target intra-market bundle arb (YES+NO ≠ $1.00) in high-liquidity conditions rather than cross-platform