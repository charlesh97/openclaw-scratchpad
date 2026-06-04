# Polymarket Bot — 4 Strategies in One Bot

**Source:** https://github.com/MrFadiAi/Polymarket-bot  
**Recommendation:** MEDIUM  
**Language:** Node.js (TypeScript)  

## What It Does

An open-source automated trading bot with 4 built-in strategies for Polymarket:
1. **DipArb** — buy the dip when prices drop abnormally
2. **Copy Trading** — mirror top leaderboard traders (60%+ win rate filter)
3. **Smart Money** — follow wallets with consistent profit history
4. **Market Making** — provide liquidity and capture spread

## Key Features

- 4-layer protection system (daily 5%, monthly 15%, drawdown 25%, total halt 40%)
- Smart money filtering (60%+ win rate, 1.5x profit factor)
- Dynamic position sizing (reduces during losses, increases during wins)
- Whale trade detection (prevents following lucky one-hit wonders)
- Dashboard UI for monitoring
- Gas fee accounting built-in

## Risks

- Node.js not ideal for HFT — latency-sensitive strategies may underperform
- Copy trading depends on leaderboard quality which degrades over time
- v3.1 is relatively new (January 2026) — limited battle testing
- Dashboard requires build step (npm run build)
- Less sophisticated signal processing vs Python/Rust alternatives

## Implementability: 3/5

Easy to set up but strategy quality is uncertain. Good for learning, but copy trading is a crowded space.

## Status: QUEUED
