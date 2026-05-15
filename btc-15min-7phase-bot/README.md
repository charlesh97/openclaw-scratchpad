# BTC 15-Minute 7-Phase Trading Bot

**Source:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot
**Recommendation:** YES

## What It Does

A production-grade algorithmic trading bot specifically designed for Polymarket's 15-minute BTC up/down price prediction markets. Uses a modular 7-phase architecture combining multiple signal sources (spike detection, sentiment analysis, price divergence) with professional risk management.

## Architecture (7-Phase Pipeline)

1. **Ingestion** - Unifies and validates data from Coinbase, Binance, News, Solana
2. **Nautilus Core** - Trading framework layer
3. **Signal Processors** - Spike detection, sentiment analysis, price divergence
4. **Fusion Engine** - Weighted voting to combine signals
5. **Risk Management** - $1 max per trade, 30% stop loss, 20% take profit
6. **Execution** - Polymarket order placement
7. **Monitoring & Learning** - Grafana dashboards, self-optimizing weight adjustment

## Key Features

- **Multi-Signal Intelligence**: Combines 3 independent signal types before trading
- **Risk-First Design**: Conservative sizing, automatic stop-loss
- **Dual-Mode Operation**: Toggle between simulation and live without restart
- **Self-Learning**: Automatically optimizes signal weights based on historical performance
- **Auto-Recovery**: WebSocket reconnection, rate limiting, data validation
- **Paper Trading**: Full P&L tracking in simulation mode

## Why It Matters

This is one of the most complete open-source bots for short-duration crypto prediction markets. The 7-phase architecture is production-grade and the self-learning component makes it adaptive to changing market conditions. After Polymarket's Feb 2026 delay removal, short-term crypto market bots need this level of sophistication.

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Signal decay over time | Medium | Self-learning weight adjustment |
| Polymarket API changes | Medium | Modular architecture, easy to swap |
| Network latency in crypto mkts | Medium | Nautilus framework handles this |
| Overfitting to historical data | Low | Dual-mode simulation allows backtesting |

## Implementability: 4/5

Well-documented Python project with clear setup instructions. Requires Nautilus Trader framework and Grafana for monitoring. The 7-phase pipeline is modular so components can be extracted and adapted.

## Next Steps

1. Clone and run in simulation mode with historical data
2. Tune signal weights for specific market conditions
3. Deploy in paper trading mode for 2+ weeks
4. Monitor signal correlation and adapt fusion engine
