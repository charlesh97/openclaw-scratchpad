# BTC 15-Minute Trading Bot — 7-Phase Production Architecture

**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot  
**Status:** Active  
**Language:** Python  
**Recommendation:** MEDIUM — Excellent architecture reference, narrow focus

## What It Does

A production-grade algorithmic trading bot specifically for Polymarket's 15-minute BTC price prediction markets. Features a 7-phase architecture combining multiple signal sources, professional risk management, and self-learning capabilities.

## Why It Matters

- **7-phase architecture** is the most sophisticated bot structure we've seen — excellent design reference
- Self-learning capabilities via adaptive parameter tuning
- Combines technical signals, on-chain data, and market microstructure
- Professional risk management: dynamic position sizing, stop-loss, max drawdown limits

## 7 Phases

1. **Data Collection** — Real-time BTC price feeds + Polymarket order book
2. **Signal Generation** — Technical indicators + on-chain metrics
3. **Probability Estimation** — ML model for fair value prediction
4. **Risk Assessment** — VaR, max drawdown, Kelly criterion sizing
5. **Execution** — CLOB API with smart order routing
6. **Monitoring** — Real-time PnL, slippage tracking, fill quality
7. **Self-Learning** — Adaptive parameter updates based on recent performance

## Implementability: 3/5

- Excellent architecture reference for our bot's next iteration
- Narrowly focused on BTC 15-min markets only
- Self-learning component is valuable but untested at scale
- Some phases overlap with our existing pipeline

## Risks

- Overfitted to BTC 15-min market patterns (may not generalize)
- ML probability estimation adds model risk + drift
- 7-phase architecture increases surface area for bugs

## Next Steps

1. Extract risk management module (dynamic sizing, drawdown control) — highest value component
2. Study signal combination methodology for our own multi-signal strategy
3. Backtest the probability estimation model on historical BTC 15-min data
