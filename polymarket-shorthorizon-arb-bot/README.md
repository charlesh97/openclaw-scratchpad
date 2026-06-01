# Polymarket Short-Horizon Arbitrage Bot

**Source:** https://github.com/PolyBullLabs/polymarket-5min-15min-1hour-arbitrage-trading-bot
**Recommendation:** MEDIUM — Educational reference for short-horizon crypto Up/Down strategies
**Implementability:** 4/5

---

## What It Does

Three production-style Python Polymarket bot implementations for short-horizon (5-minute, 15-minute, and 1-hour) crypto Up/Down prediction markets. Ships with WebSocket market data integration, CLOB-style order flow patterns, and a Jupyter notebook for live market analysis.

## Bots Included

### Bot 1: 5-Minute Market-Neutral Strategy
- Trades on very short timeframes (market resolution every 5 minutes)
- Targets small edges with high frequency
- Uses limit orders to capture spread

### Bot 2: 15-Minute Trend-Following Strategy
- Medium-frequency strategy on 15-minute markets
- Uses moving average crossovers and momentum indicators
- Includes dynamic position sizing based on confidence

### Bot 3: 1-Hour Statistical Edge Strategy
- Slest frequency, higher conviction per trade
- Combines multiple signal types: price action, volume profile, order book imbalance

## Key Features

| Feature | Details |
|---------|---------|
| **Language** | Python 3.9+ |
| **Data** | WebSocket real-time market data |
| **Execution** | CLOB API via `py-clob-client` |
| **Analysis** | Jupyter notebook included |
| **Risk** | Configurable per-strategy parameters |
| **Documentation** | English + Simplified Chinese |

## Trading Strategies

### Bot 1 — Market-Neutral (5-min)
```
Entry: YES + NO sum deviation from $1.00
Signal: Combined cost < $0.98
Exit: Auto-resolve at market end
Position: Fixed size per trade
```

### Bot 2 — Trend Following (15-min)
```
Entry: Price crossing EMA(10) above/below SMA(20)
Signal: Confluence with volume surge > 2x average
Exit: Take profit at 80% confidence or stop-loss
Position: Kelly-sized based on signal strength
```

### Bot 3 — Statistical Edge (1-hour)
```
Entry: Multiple indicator convergence (RSI + MACD + order book)
Signal: Z-score > 2.0 on price deviation from volume-weighted average
Exit: Regression to mean or time-based expiry
Position: Scaled by conviction score
```

## Why This Matters

- **Educational** — Clean, well-documented Python code that's easy to learn from
- **Short-horizon focus** — Up/Down 5min/15min/1hour markets are the most active on Polymarket
- **WebSocket design** — Shows proper real-time data handling patterns
- **Clone-and-run** — Minimal setup, works with Polymarket's standard API

## Risks

- Educational/experimental — not optimized for production
- Short-horizon markets are extremely competitive (HFT bots dominate)
- Polymarket recently introduced dynamic fees that may affect profitability
- No documented PnL track record

## Next Steps

1. Clone and run in dry-run/simulation mode
2. Analyze performance on 1-hour markets (least competitive timeframe)
3. Backtest against our own arb strategies to compare approaches
4. Port the WebSocket data handling into our arb-bot-analysis framework
