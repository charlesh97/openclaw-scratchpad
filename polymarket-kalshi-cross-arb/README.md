# Polymarket-Kalshi Cross-Platform Arbitrage Bot (realfishsam)

**Source:** https://github.com/realfishsam/prediction-market-arbitrage-bot

## What It Does
A bot that detects and executes arbitrage strategies between Polymarket and Kalshi. Auto-buys low on one platform, sells high on the other. Built with https://pmxt.dev.

## Key Features
- Real-time price monitoring across both platforms
- Cross-platform order execution (buy on Polymarket, sell on Kalshi)
- Built on pmxt.dev SDK for unified API access
- Simple buy-low/sell-high logic

## Implementability: 4/5
**MEDIUM** — clean, simple implementation. pmxt.dev SDK abstracts cross-platform complexity. Main challenge: holding periods during cross-platform settlement require capital to be locked up while arb converges.

## Next Steps
1. Test on the BTC 1H market (most liquid for cross-platform arb)
2. Add slippage estimation (current version might underestimate execution cost)
3. Determine minimum profitable spread accounting for both gas and holding risk
