# Polymarket 5-Strategy Trading Bot (PolyHFT)

**Source:** https://github.com/Anmoldureha/polymarket-trading-bot-strategies

## What It Does

An advanced algorithmic trading bot for Polymarket with **10 trading strategies** (1 beta):
1. **Hedging** — Cross-exchange hedging with Hyperliquid
2. **Micro-spreads** — High-frequency spread capture in low-priced markets ($0.05-$0.10)
3. **Liquidity Provision** — Market making with liquidity rewards (up to $50/market)
4. **Single-Market Arbitrage** — YES+NO bundle mispricing
5. **Low-Volume Opportunities** — Thinly traded market exploitation
6. **Spread Scalping** — Bid-ask spread capture
7. **Tail-End Trading** — Extreme probability plays
8. **Combinatorial Arbitrage** — Multi-market arbitrage detection
9. **Legged Arbitrage** — Multi-leg arbitrage execution
10. **Continuous Market Making** [BETA] — Ongoing two-sided quoting

### Key Features
- Enterprise risk management (position limits, stop-losses, drawdown)
- Parallel market fetching with 5-second cache TTL (90% API call reduction)
- Telegram notifications
- State persistence for seamless restarts

## Implementability: 4/5

Python, well-structured, YAML config. The combinatorial arbitrage module is directly applicable to our arb bot.

## Recommendation: MEDIUM

Excellent breadth of strategies. The combinatorial arbitrage and micro-spread strategies are particularly valuable. Worth queuing for deeper integration.
