# BTC 15-Minute Prediction Market Trading Bot

**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot

## What it does

A production-grade algorithmic trading bot for Polymarket's 15-minute BTC price prediction markets. Features a 7-phase architecture combining multiple signal sources (technical analysis, on-chain data, sentiment), professional risk management, and self-learning capabilities.

## Architecture (7-Phase)

1. **Data Ingestion:** Real-time BTC price feeds, order book data
2. **Signal Generation:** Technical indicators + on-chain metrics
3. **Probability Estimation:** ML model predicting short-term price direction
4. **Position Sizing:** Kelly criterion-based allocation
5. **Execution:** Smart order routing to minimize slippage
6. **Risk Management:** Stop-loss, max drawdown limits
7. **Self-Learning:** Performance tracking and strategy adaptation

## Why it matters

The 15-minute crypto markets are among the most liquid on Polymarket. This bot provides a complete, extensible framework specifically for these high-frequency binary markets. Multi-signal approach is more robust than single-signal strategies.

## Implementability: 4/5

Complete codebase but requires BTC price feed infrastructure and ML model training. The self-learning component is sophisticated.

## Risks

- 15-minute markets resolve quickly — latency is critical
- ML model needs regular retraining
- Polymarket dynamic fees designed to reduce latency arb advantage
- High competition from other bots

## Next Steps

1. Set up BTC price feed (Binance WebSocket)
2. Train initial probability model on historical data
3. Enable dry-run mode for 2 weeks
4. Calibrate position sizing to Polymarket's fee structure
