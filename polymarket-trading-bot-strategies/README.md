# Polymarket Trading Bot Strategies (PolyHFT)

**Source:** https://github.com/Anmoldureha/polymarket-trading-bot-strategies  
**Author:** Anmoldureha  
**Language:** Python  
**License:** MIT  
**Status:** Active

## What It Does

A comprehensive algorithmic trading bot implementing **10 distinct trading strategies** for Polymarket. Covers the full spectrum of prediction market trading approaches:

1. **Hedging** — Offset positions across correlated markets
2. **Micro-Spreads** — Capture tiny bid-ask spreads at high frequency
3. **Liquidity Provision** — Earn spread by providing limit orders
4. **Single-Market Arbitrage** — YES+NO bundle detection
5. **Low-Volume Opportunities** — Exploit illiquid market mispricings
6. **Spread Scalping** — Quick in-and-out on spread movements
7. **Tail-End Trading** — Long-shot deviation betting
8. **Combinatorial Arbitrage** — Cross-condition dependency detection
9. **Legged Arbitrage** — Multi-leg synthetic positions
10. **Continuous Market-Making [BETA]** — Persistent two-sided quoting

## Why It Matters

- **Most comprehensive single repo** — Covers almost every known Polymarket strategy
- **Enterprise risk management** — Multi-layered risk controls with position limits, stop-losses, drawdown protection
- **Paper trading mode** — Safe simulation before real capital
- **Telegram notifications** — Real-time trade alerts
- **Parallel market fetching** — Optimized for speed

## Risks

- Breadth over depth — each strategy may need individual tuning
- Some strategies cannibalize each other (e.g., MM vs arb)
- Beta strategies may be unproven
- High strategy count = high maintenance surface

## Implementability: 3/5

- Well-structured Python codebase
- Comprehensive but complex configuration
- Best used as a strategy library — pick specific strategies, not all 10
- Heavy API usage may hit rate limits

## Next Steps

1. Extract specific strategies of interest (combinatorial arb, micro-spreads)
2. Test in paper trading mode
3. Validate strategy independence and interaction effects
