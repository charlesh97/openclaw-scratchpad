# Polymarket BTC 15-Minute Trading Bot

**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot

## What it does
A production-grade algorithmic trading bot for Polymarket's 15-minute BTC price direction prediction markets. Uses a **7-phase architecture** combining multiple signal sources (technical indicators, on-chain data, sentiment), professional risk management, and self-learning capabilities.

## Architecture (7 Phases)
1. **Data ingestion** — Real-time BTC price feeds + on-chain metrics
2. **Signal generation** — Multiple indicator ensemble (RSI, MACD, volume profile, order flow)
3. **Probability estimation** — Converts signals to outcome probabilities
4. **Position sizing** — Kelly criterion with volatility adjustment
5. **Execution** — Smart order routing to minimize slippage
6. **Monitoring** — Real-time P&L, drawdown, exposure tracking
7. **Self-learning** — Periodic retraining on new market data

## Why it matters
This is the most architecturally complete Polymarket trading bot found. The 7-phase structure is a template for any serious prediction market bot. The BTC 15-min markets are among the most liquid on Polymarket.

## Implementability: 4/5
Python + standard ML stack. Clearly documented. The self-learning component adds complexity but the core signal + execution pipeline is straightforward to adapt.

## Next Steps
1. Clone and study the 7-phase architecture
2. Adapt signal generation for non-BTC markets
3. Integrate with existing arb detection logic
