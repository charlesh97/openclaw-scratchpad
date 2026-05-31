# Polymarket Trading Bot Strategies (5-Strategy Suite)

**Source:** https://github.com/Anmoldureha/polymarket-trading-bot-strategies

## What it does
An advanced algorithmic trading bot for Polymarket with **5 distinct strategies**: hedging, micro-spreads, liquidity provision, arbitrage, and low-volume opportunities. Includes comprehensive risk management, paper trading mode, and Telegram notifications.

## The 5 Strategies
1. **Hedging** — Opens offsetting positions across related markets to lock in profit
2. **Micro-spreads** — Captures very small bid-ask spreads at high frequency
3. **Liquidity Provision** — Acts as market maker, earning the spread
4. **Arbitrage** — Cross-market and bundle arbitrage detection
5. **Low-Volume Opportunities** — Exploits mispricing in illiquid markets

## Why it matters
Having all 5 strategies in one codebase provides a complete trading system to study. The multi-strategy approach is exactly what we want for our arb bot — combining arb with liquidity provision and hedging creates a more robust portfolio.

## Implementability: 3/5
The codebase is TypeScript (Node.js). Adapting to our Python stack requires a port. But the strategy logic is well-documented and the risk management framework is directly transferable.

## Next Steps
1. Study the risk management system
2. Port the 5 strategy templates to Python
3. Add cross-exchange arb (Polymarket ↔ Kalshi)
