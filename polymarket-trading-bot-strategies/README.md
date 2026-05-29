# Polymarket Trading Bot Strategies (PolyHFT)

**Source:** https://github.com/Anmoldureha/polymarket-trading-bot-strategies
**Recommendation:** MEDIUM (strong, but already many similar bots exist)
**Implementability:** 4/5
**Last updated:** ~May 2026

## What it does

A comprehensive algorithmic trading bot with **10 distinct trading strategies** including hedging, micro-spreads, liquidity provision, single-market arbitrage, low-volume opportunities, spread scalping, tail-end trading, combinatorial arbitrage, legged arbitrage, and continuous market-making.

### Key features:
- **10 strategies** running simultaneously with independent risk controls
- **Cross-exchange hedging** with Hyperliquid
- **YAML-based configuration** for easy strategy tuning
- **State persistence** — auto-saves and restores state on restart
- **Telegram notifications** for trade alerts
- **Paper trading mode** for safe testing

## Why it matters

The breadth of strategies is impressive — particularly the combinatorial arbitrage detection and cross-exchange hedging with Hyperliquid. Good reference for strategy diversity.

## Risks

- Strategy count may cause resource contention
- Documentation is extensive but some strategies are labeled BETA
- Hyperliquid integration adds dependency risk

## Next Steps
1. Test the 5 core strategies in paper trading
2. Focus on combinatorial arbitrage strategy (most differentiated)
3. Evaluate cross-exchange hedging utility
