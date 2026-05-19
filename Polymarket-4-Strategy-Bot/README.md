# Polymarket 4-Strategy Trading Bot

**Source:** [MrFadiAi/Polymarket-bot](https://github.com/MrFadiAi/Polymarket-bot)
**Recommendation:** MEDIUM

## What It Does

An open-source Node.js bot implementing 4 strategies for Polymarket:

1. **Copy Trading** — Follow top-performing wallets (60%+ win rate filter)
2. **Dip Arbitrage (DipArb)** — Buy during panic sell-offs
3. **Smart Money** — Filtered whale monitoring
4. **Auto-Rotation** — Dynamic strategy switching

### Risk Management
- 4-layer protection: Daily (5%), Monthly (15%), Drawdown (25%), Total Loss Halt (40%)
- Dynamic position sizing
- Whale trade detection (prevents following lucky one-hit wonders)
- Gas fee accounting

## Implementability: 3/5

- Well-documented with active development (v3.1 Jan 2026)
- Node.js — easy to deploy
- Strategy auto-rotation is interesting

## Risks
- Copy trading depends on finding genuine alpha traders (may be declining)
- Risk management is good but may be too conservative
- Bot performance depends on quality of trader filtering

## Next Steps
Evaluate copy trading thesis — has the edge been competed away? Consider adapting DipArb for our own strategy.
