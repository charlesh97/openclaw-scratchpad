# Polymarket-Kalshi BTC Arbitrage Bot

**Source:** https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot  
**Author:** CarlosIbCu  
**Language:** Python (FastAPI) + Next.js (dashboard)  
**License:** MIT  
**Status:** Active

## What It Does

A focused arbitrage bot that monitors **BTC 1-Hour Price prediction markets** on both Polymarket and Kalshi simultaneously, detecting risk-free opportunities when the combined cost of opposing positions is less than $1.00.

### Key Features
- Real-time price fetching every 1 second from both platforms
- Smart event matching (aligns Polymarket events with corresponding Kalshi markets)
- Multiple strategy detection (Poly Down + Kalshi Yes, Poly Up + Kalshi No)
- Includes a detailed arbitrage thesis document explaining the mathematics

## Why It Matters

- **Focused scope** — BTC hourly markets are among the most liquid on both platforms
- **Real-time dashboard** — Beautiful Next.js/shadcn UI for monitoring
- **Educational thesis** — Includes a thorough mathematical explanation of binary option arbitrage
- **Lightweight** — Simple stack, easy to understand and modify

## Risks

- Same cross-platform risks (dual accounts, dual capital)
- BTC hourly markets are competitive — latency matters
- Kalshi API may have rate limits for 1-second polling

## Implementability: 4/5

- Clean, well-structured code
- Python backend + Next.js frontend
- Excellent documentation including the thesis
- Narrow focus is a feature, not a bug

## Next Steps
1. Review the arbitrage thesis for our strategy documentation
2. Run against historical data to verify edge persistence
3. Consider extending to ETH hourly markets
