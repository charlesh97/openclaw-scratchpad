# Manifold Markets Market-Making Bot

**Source:** https://github.com/manifoldmarkets/market-maker

## What It Does
Official market-making bot for Manifold's prediction markets. Places passive limit orders based on exponential moving averages and variances of market probabilities, capturing the bid-ask spread by buying low and selling high as markets fluctuate.

## How It Works
1. Computes an **exponential moving average (EMA)** of the market probability
2. Computes an **exponential moving variance (EMV)** to gauge volatility
3. Places symmetric limit orders above and below the current market price
4. When volatility causes fills on both sides, the bot profits from the spread

## Key Features
- Simple, well-documented TypeScript codebase
- Uses official Manifold API
- Automatically provides liquidity to markets
- Configurable via .env (API key + username)

## Why It Matters
This is the **official** Manifold market maker — written by the platform team themselves. It's a production-grade reference for how basic market making works in prediction markets using statistical volatility estimates rather than complex order book models.

## Risks
- Only works on Manifold (not Polymarket/Kalshi)
- Mana-based (play money) — not real capital markets
- Requires understanding of EMA/EMV parameter tuning
- Thinly traded markets may not provide enough fills

## Implementability: 5/5
Extremely simple — TypeScript, 50 lines of core logic, clear .env configuration.

## Next Steps
1. Fork and adapt to use Polymarket/Kalshi CLOB API
2. Replace Mana with USDC settlement
3. Tune EMA windows for crypto prediction market characteristics
