# Prediction Market Arbitrage Bot — Synthetic Arb (Educational)

**Source:** https://github.com/realfishsam/prediction-market-arbitrage-bot
**Recommendation:** NO — educational only, ignores fees/slippage

## What it does

An educational Node.js bot that detects "synthetic arbitrage" between Polymarket and Kalshi. Uses pmxt.dev unified API.

### Synthetic Arbitrage
Buys YES on one platform and NO on another for the same outcome:
- Polymarket: Kevin Warsh YES = 41¢
- Kalshi: Kevin Warsh NO = 57¢
- Total cost: 98¢ → Guaranteed $1.00 = 2¢ profit

### Features
- Fuzzy matching (Jaccard + Levenshtein) for cross-platform outcome pairing
- YOLO (all-in) and CONSERVATIVE (fixed amount) trading modes
- Market order execution (aggressive liquidity taking)

## Why NO Recommendation
The readme itself warns: ignores gas fees, trading fees, and slippage. These would erode the tiny edges. Pure educational reference.

## Implementability: 3/5
Clean Node.js code, but not production-viable as-is. Good reference for cross-platform matching logic.

## Next Steps
1. Reference the fuzzy matching algorithm for market pair detection
2. Incorporate realistic fee/slippage calculations for accurate edge estimation
