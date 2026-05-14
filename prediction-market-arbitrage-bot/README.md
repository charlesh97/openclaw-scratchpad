# Prediction Market Arbitrage Bot — Polymarket ↔ Kalshi Cross-Platform Arbitrage

**Source:** https://github.com/realfishsam/prediction-market-arbitrage-bot  
**Status:** Active  
**Language:** Python (pmxt.dev-based)  
**Recommendation:** ✅ YES — Top Pick for Email

## What It Does

A cross-platform arbitrage bot that identifies and executes price discrepancies between **Polymarket** and **Kalshi** for identical prediction market contracts. Built on top of the [pmxt.dev](https://pmxt.dev) unified prediction market API.

## Why It Matters

- **Cross-platform arbitrage** is the highest-alpha opportunity in prediction markets right now
- Polymarket and Kalshi frequently diverge on identical events (BTC 1-hour, political events, sports) due to different user bases, liquidity, and fee structures
- The pmxt.dev abstraction layer handles both platform APIs, meaning one integration gives you both venues
- Directly addresses a gap in our current bot (which only targets Polymarket)

## Architecture

```
prediction-market-arbitrage-bot/
├── scanner.py         — Watches 10,000+ markets across both platforms
├── arb_calculator.py  — Computes risk-free return after fees, slippage
├── executor.py        — Places simultaneous hedge orders on both platforms
├── config.yaml        — Fee thresholds, min profit, capital allocation
└── dashboard/         — Real-time opportunity display
```

## How Cross-Platform Arbitrage Works

1. **Scan:** Watch both Polymarket and Kalshi for the same event/market pair
2. **Compare:** If Polymarket YES = 42¢ and Kalshi YES = 48¢ → buy Polymarket, sell Kalshi
3. **Hedge:** The positions offset because both resolve to the same outcome
4. **Net profit:** 6¢ minus fees (typically 1-2% per side)

## Risks

- **Execution lag:** The 500ms delay between detection and execution on two separate platforms can cause partial fills
- **Kalshi restrictions:** Kalshi has KYC requirements and position limits for non-accredited users
- **Fee erosion:** Taker fees on both platforms can wipe out thin spreads (1-2% per leg)
- **API rate limits:** Both platforms enforce rate limits that reduce scan throughput

## Implementability: 4/5

- Well-structured Python code with clear separation of scanner/calculator/executor
- pmxt.dev abstracts API differences — one integration point
- Requires accounts on both Polymarket (crypto wallet) and Kalshi (KYC'd USD account)
- Risk-free in theory but execution risk in practice needs careful modeling

## Next Steps

1. Review pmxt.dev API for current platform coverage and rate limits
2. Backtest historical price differences between Polymarket and Kalshi BTC markets
3. Model fee impact and minimum viable spread thresholds
4. Consider extending to Robinhood Prediction Markets (launching 2026)
