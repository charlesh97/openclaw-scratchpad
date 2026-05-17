# Polymarket 4-Strategy Trading Bot

**Source:** https://github.com/MrFadiAi/Polymarket-bot  
**Recommendation:** MEDIUM

## What It Does

An open-source trading bot that implements 4 distinct strategies for Polymarket in a single Node.js application. Features a 4-layer protection system (daily 5%, monthly 15%, drawdown 25%, total loss 40%) and smart money filtering that only follows traders with 60%+ win rate and 1.5x profit factor.

## Strategies

1. **Copy Trading** — Mirrors profitable traders with smart filtering
2. **Dip Arbitrage** — Buys YES positions on temporary price dips
3. **Market Making** — Provides liquidity on both sides of the book
4. **Event-Driven** — Trades specific event outcomes based on triggers

## Key Features

- 4-layer risk management protection system
- Smart money filtering (60%+ win rate, 1.5x profit factor)
- Dynamic position sizing (reduces during losses, increases during wins)
- Whale trade detection (prevents following lucky one-hit wonders)
- Gas fee accounting with higher profit thresholds
- Real-time risk dashboard

## Why It Matters

Well-maintained project with regular updates (v3.1 January 2026). The multi-strategy approach diversifies risk. Strong documentation and setup guides.

## Implementability: 4/5

Node.js, well-documented, easy to set up. Requires MetaMask wallet and private key. Good for intermediate developers.

## Risks

- Copy trading depends on finding truly skilled traders
- Overlap between strategies could compound risk
- Gas fees on Polygon still non-trivial for small positions
