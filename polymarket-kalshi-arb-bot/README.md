# Polymarket ↔ Kalshi Arbitrage Bot

**Source:** https://github.com/realfishsam/prediction-market-arbitrage-bot

## What it does
A bot that continuously monitors price differences between Polymarket and Kalshi for identical prediction market events. When the same outcome is priced differently on the two platforms, the bot automatically buys the underpriced side on one exchange and sells (or holds to resolution) the overpriced side on the other, locking in a risk-free profit.

Built on top of `pmxt.dev` — a cross-platform prediction market exchange toolkit.

## Why it matters
- **Risk-free arbitrage**: By definition, identical events must resolve to the same payout. Any price divergence is pure edge.
- **Massive addressable market**: The Polymarket/Kalshi spread was a $40M+ profit pool between April 2024 and April 2025.
- **Cross-exchange infrastructure**: Demonstrates how to aggregate liquidity across fragmented prediction markets — a key capability for any serious arb bot.

## Architecture
- **Data layer**: Subscribes to real-time order book feeds from both Polymarket (CTF/neg-risks) and Kalshi (REST API).
- **Detection engine**: Scans for events listed on both platforms, normalizes outcome names, computes implied probabilities, flags divergences > threshold + fees.
- **Execution module**: Places concurrent orders on both sides — market or limit depending on urgency.
- **Risk controls**: Max position size per event, minimum spread %, stop-loss on failed fills.

## Risks
- **Execution risk**: One leg fills, the other doesn't → directional exposure.
- **Withdrawal/Settlement timing**: Kalshi and Polymarket settle on different schedules.
- **Fees**: Polymarket has maker/taker fees; Kalshi has transaction fees. Must be modeled precisely.
- **Liquidity**: Larger positions may not fill without moving the market.

## Implementability: 4/5
The codebase is Python, well-documented, and uses pmxt.dev which abstracts exchange-specific API quirks. The core logic is straightforward cross-exchange arbitrage. Biggest challenge is maintaining up-to-date API connectivity as both platforms evolve.

## Next Steps
1. Clone repo and set up API keys for both Polymarket (via privy/email) and Kalshi.
2. Run in paper-trading mode to validate spread detection.
3. Tune minimum spread threshold and position sizing.
4. Deploy on a low-latency VPS near both exchange API endpoints.
