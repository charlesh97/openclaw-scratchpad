# Polymarket Arbitrage Bot — Cross-Platform (10,000+ Markets)

**Source:** https://github.com/ImMike/polymarket-arbitrage
**Recommendation:** MEDIUM — good cross-platform arb detection, but real arb is rare

## What it does

A Python-based arbitrage bot that scans **5,000+ Polymarket and Kalshi markets** looking for price inefficiencies. Supports three detection modes:

1. **Cross-Platform Arbitrage**: Detects price differences between Polymarket and Kalshi for identical predictions
2. **Bundle Arbitrage**: YES + NO prices that don't sum to ~$1.00
3. **Market Making**: Captures spreads via competitive bid/ask orders

Includes a live web dashboard, dual data modes (real/simulation), and market matching AI using text similarity.

## Key Finding
99.6% win rate in simulation ($573 profit) — but real markets are highly efficient. The bot acknowledges this honestly.

## Implementability: 4/5
Clean Python architecture, well-documented YAML config, dashboard included.

## Next Steps
1. Run in simulation mode to understand arb patterns
2. Identify which market categories show the most persistent inefficiencies
3. Deploy with conservative capital in sports markets during live events
