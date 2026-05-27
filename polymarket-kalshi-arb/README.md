# Polymarket-Kalshi Arbitrage Bot

**Source:** https://github.com/ImMike/polymarket-arbitrage

## What It Does

A Python arbitrage bot that watches **10,000+ markets** across Polymarket and Kalshi, looking for price inefficiencies.

### Arbitrage Detection Types
- **Cross-Platform Arbitrage**: Price differences between Polymarket and Kalshi for the same prediction
- **Bundle Arbitrage**: YES + NO prices that don't sum to ~$1.00
- **Market Making**: Spread capture via competitive bid/ask orders

### Features
- Dual data modes (real or simulation)
- Market Matching AI — automatically matches similar predictions across platforms using text similarity
- Live web dashboard
- Fee accounting (gas costs included)
- Risk management with kill switch

## Implementability: 5/5

Clean Python codebase, excellent documentation, easy to configure. The cross-platform approach is directly applicable to our arb bot project.

## Recommendation: MEDIUM

Strong implementation. The cross-platform scanning and market matching AI are valuable additions to our toolkit.
