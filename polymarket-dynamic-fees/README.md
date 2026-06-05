# Polymarket Dynamic Taker Fees — Market Structure Change

**Source:** https://www.financemagnates.com/cryptocurrency/polymarket-introduces-dynamic-fees-to-curb-latency-arbitrage-in-short-term-crypto-markets/  
**Type:** Platform policy change  
**Date:** January 2026  
**Added:** 2026-06-05

## What Changed

Polymarket introduced a **dynamic taker-fee model** on its 15-minute crypto markets. Key details:

- Taker fees apply only to 15-minute crypto markets (BTC, ETH, SOL, XRP)
- Fees are **highest near 50/50 odds** — up to ~3.15% on a 50¢ contract
- Fees fund the **Maker Rebates Program** — redistributed daily to liquidity providers
- Other markets (longer-dated, non-crypto) remain fee-free

## Why It Matters for Algorithmic Trading

This is a **major structural change** for bot strategies:

1. **Latency arb is dead** — The classic "catch price dislocations between Polymarket and spot" strategy was generating $313→$414k in one month for one wallet. The new fee structure exceeds typical arb margins at 50/50 odds.

2. **Maker strategy is now viable** — The rebate program rewards two-sided order placement. Poly-Maker (see research folder) was built exactly for this.

3. **Strategy rotation required** — Bots that relied purely on taker-side latency arb need to pivot to:
   - Cross-platform arb (Polymarket ↔ Kalshi)
   - Market-making (collect maker rebates)
   - Longer-dated markets (fee-free)
   - Combinatorial arb (cross-market dependencies)

4. **Barrier to entry rises** — HFT latency strategies are now uneconomical. Deeper analysis strategies (ML, combinatorial) become relatively more attractive.

## Implementability: 5/5 (as research context)

Not a trading bot — this is critical **market intelligence**. Every arbitrage strategy must account for these fees in P&L calculations.

## Impact on Our Strategy Pipeline

| Strategy | Before Dynamic Fees | After Dynamic Fees |
|----------|-------------------|-------------------|
| 15-min bundle arb (YES+NO < $1) | Profitable | ~3.15% fee kills thin margins |
| Cross-platform arb | Unaffected | Still valid |
| Market making (maker rebates) | Moderate | Highly incentivized |
| Longer-date arb strategies | Unaffected | Unaffected |
| Combinatorial arb | Unaffected | Unaffected |
