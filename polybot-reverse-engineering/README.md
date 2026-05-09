# polybot — Reverse-Engineering Toolkit + AWARE Foundation

**Source:** https://github.com/ent0n29/polybot
**Recommendation:** MEDIUM (infrastructure quality, not a standalone strategy)

## What it does

An **open-source Polymarket trading infrastructure and strategy reverse-engineering toolkit**. Rather than a single strategy bot, polybot provides:

- **Market data capture:** Real-time order book and trade data ingestion from Polymarket's API
- **Strategy reverse-engineering module:** Analyzes the behavioral patterns of successful traders on Polymarket to reverse-engineer their positioning logic
- **AWARE integration:** Powers `AWARE` — a trader intelligence product that builds PSI (Polymarket Smart Index) indices and offers fund mirroring

This is execution + data infrastructure, not a single-strategy arb bot.

## Why it matters

- Provides the **market microstructure data** needed to actually evaluate whether an arbitrage opportunity is real and executable at a given moment
- AWARE's PSI indices aggregate the trading behavior of smart money wallets — useful for identifying when the "smart money" is positioning for a specific outcome, which can inform directional bias in arbitrage sizing
- The strategy reverse-engineering component could surface non-obvious correlations between Polymarket prices and external data feeds that could be exploited

## Risks

- This is infrastructure, not a finished strategy — requires significant integration work to turn it into a profit-generating bot
- AWARE is a commercial product; polybot is the open-source subset
- No guaranteed uptime or API rate limit compliance — Polymarket can change their API without notice

## Implementability: 3/5

- Great data pipeline to start from — saves 3-6 months of building order book capture infrastructure
- Requires Python + Polymarket API integration + database layer
- Strategy reverse-engineering module needs ML pipeline (not clearly documented in search results)

## Next Steps

1. Clone and build the market data capture pipeline first
2. Use it to build a historical order book dataset for backtesting
3. Evaluate the strategy reverse-engineering module against known profitable wallets
4. Use AWARE PSI index signals as an input to position sizing in existing arb strategies

---
*Part of vega's arb-bot-analysis research.*