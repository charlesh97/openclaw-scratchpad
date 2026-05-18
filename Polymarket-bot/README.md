# Polymarket-bot — 4 Strategies in One Bot

**Source:** https://github.com/MrFadiAi/Polymarket-bot
**Recommendation:** ✅ YES — production-ready, actively maintained, great risk management

## What it does

A complete Node.js trading bot for Polymarket with **4 integrated strategies** and a web dashboard. Created by @Mr_CryptoYT, actively maintained (v3.1, Jan 2026).

### 4 Strategies
1. **Dip Arbitrage (DipArb)**: Buys YES tokens when price drops significantly below fair value; sells into rebounds
2. **Copy Trading**: Mirrors top Polymarket traders with smart filtering (60%+ win rate, 1.5x profit factor)
3. **Market Making**: Two-sided quoting to capture bid-ask spread
4. **Cross-Platform Arbitrage**: Detects price differences between related Polymarket markets

### Risk Management (4-Layer Protection)
- Daily max loss: 5%
- Monthly max loss: 15%
- Drawdown halt: 25% from peak
- Total loss permanent halt: 40%
- Smart Money Filtering: minimum $1.50 per DipArb position (guarantees exit)

## Implementability: 5/5
Node.js, simple setup, excellent documentation, web dashboard, risk management built-in. Best option for getting started quickly.

## Next Steps
1. Clone and run in dry-run mode
2. Configure Telegram alerts
3. Test DipArb strategy on 15-min BTC/ETH markets
4. Gradually add more strategies
