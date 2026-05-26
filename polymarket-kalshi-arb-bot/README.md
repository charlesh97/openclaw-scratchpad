# Polymarket-Kalshi Arbitrage Bot

**Source:** https://github.com/ImMike/polymarket-arbitrage  
**Author:** ImMike  
**Language:** Python  
**License:** MIT  
**Status:** Active

## What It Does

A cross-platform arbitrage bot that simultaneously monitors 5,000+ Polymarket and Kalshi markets, detecting price discrepancies between the two platforms for identical or equivalent prediction events. It executes three distinct strategies:

1. **Cross-Platform Arbitrage** — Buys YES on Kalshi at X cents, buys equivalent position on Polymarket at Y cents, locks in risk-free profit when X+Y < $1.00
2. **Bundle Arbitrage** — Detects when YES + NO prices on a single market don't sum to ~$1.00
3. **Market Making** — Captures spreads by placing competitive bid/ask orders on both platforms

## Why It Matters

- **Largest addressable market** — Kalshi (CFTC-regulated) + Polymarket (decentralized) = most events covered
- **Proven performance** — 99.6% win rate in simulation, $573 simulated profit
- **Live dashboard** — Real-time web UI showing opportunities and bot activity
- **Fee-aware** — Accounts for gas costs and platform fees in edge calculations
- **Dual data modes** — Switch between real market data and simulation without restart

## Risks

- Cross-platform arbitrage requires capital on both platforms simultaneously
- Kalshi is US-regulated (CFTC); Polymarket is decentralized — regulatory landscape differs
- Fast execution needed; opportunities can vanish in seconds
- Gas costs on Polygon can eat into small edges

## Implementability: 4/5

- Clean Python codebase with FastAPI backend
- Next.js dashboard for real-time monitoring
- Well-documented vs config
- Need accounts on both Polymarket and Kalshi
- Dual data modes make testing safe

## Next Steps

1. Run in simulation mode to verify edge calculations
2. Fund both platforms with target capital
3. Start with $0.03+ edge filter, scale down
4. Monitor dashboard for opportunity frequency and size
