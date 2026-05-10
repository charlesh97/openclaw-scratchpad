# MrFadiAi/Polymarket-bot

**Source:** https://github.com/MrFadiAi/Polymarket-bot
**Type:** Multi-strategy Polymarket trading bot
**Recommendation:** YES

## What It Does

All-in-one Polymarket bot with 4 integrated trading strategies, actively maintained with major updates in late 2025 and early 2026:

### v3.1 (January 2026)
- Enhanced Risk Management — position-level and portfolio-level risk controls
- Smart Money improvements — detects and tracks whale/smart money order flow as signal
- Dynamic Position Sizing — adjusts bet size based on confidence and volatility

### v3.0 (December 2025)
- Dashboard for monitoring all strategies
- Multi-strategy support with auto-rotation between strategies based on market conditions

## Why It Matters

This is the most actively maintained Polymarket bot found today, with major releases as recently as January 2026. The combination of:
- Multi-strategy approach (not just arb)
- Smart Money detection (whale tracking as alpha signal)
- Dynamic sizing (risk-responsive position management)

...makes it a more sophisticated system than simple arb bots.

## Implementability: 4/5

Production-ready Python codebase. Well-documented, actively developed. The Smart Money detection layer adds an alpha component beyond pure price arbitrage.

**Next steps:**
1. Clone and run in paper-trade mode
2. Study the Smart Money detection logic
3. Assess the dynamic sizing algorithm
4. Integrate the risk management module into your own system