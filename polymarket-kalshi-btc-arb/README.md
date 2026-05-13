# Polymarket-Kalshi BTC Arbitrage Bot

**Source:** https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot

## What it does

A real-time arbitrage bot for Bitcoin 1-Hour Price markets between Polymarket and Kalshi. Leverages Polymarket's CLOB and Kalshi's REST API to calculate combined cost of opposing positions (e.g., "Yes" on Kalshi + "Down" on Polymarket) for the same hourly expiration.

## How it works

1. Fetches BTC 1-Hour market data from both platforms
2. Calculates arbitrage: buy "Down" on one, buy "Up" on the other
3. If combined cost < $1.00, risk-free profit exists
4. Executes simultaneous orders on both platforms

## Why it matters

BTC hourly markets are among the most traded on Polymarket and Kalshi. The hourly resolution window provides frequent, predictable arbitrage opportunities. Both platforms must resolve to $1.00 for the correct outcome, so pricing discrepancies are genuine arb.

## Implementability: 4/5

Well-defined scope (BTC hourly only) makes this straightforward to deploy. Python-based with clear API integrations.

## Risks

- Requires Kalshi account (US KYC)
- Hourly markets have competition from other bots
- Polymarket dynamic fees (introduced 2026) may reduce edges
- Platform downtime could cause imbalanced positions

## Next Steps

1. Set up Polymarket + Kalshi API credentials
2. Deploy on dedicated server with low-latency Polygon RPC
3. Run paper trading for 1 week
4. Monitor dynamic fee impact on profitability
