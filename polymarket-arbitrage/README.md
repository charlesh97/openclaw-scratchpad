# Polymarket & Kalshi Cross-Platform Arbitrage Bot

**Source:** https://github.com/ImMike/polymarket-arbitrage
**Recommendation:** MEDIUM
**Implementability:** 5/5
**Last updated:** ~2026

## What it does

A Python bot that scans 5,000+ live Polymarket markets and detects arbitrage opportunities across Polymarket and Kalshi. Includes bundle arbitrage detection (YES+NO ≠ $1.00) and market making strategies.

### Key features:
- Cross-platform arbitrage (Polymarket ↔ Kalshi)
- Bundle arbitrage detection (YES + NO price ≠ $1.00)
- Market making (bid/ask spread capture)
- Live dashboard at localhost:8000
- 99.6% win rate in simulation mode
- Market Matching AI using text similarity

## Why it matters

Clean, well-documented Python implementation. The cross-platform arb between Polymarket and Kalshi is especially relevant as these are the two largest prediction markets. The Market Matching AI for cross-platform pair identification is clever.

## Risks
- Real markets are highly efficient — arb opportunities are rare
- Simulation win rate (99.6%) is misleading for live trading
- Kalshi API may have rate limits

## Next Steps
1. Run in simulation mode to understand strategy mechanics
2. Connect real API keys for both platforms
3. Focus on bundle arbitrage on Polymarket alone first (simpler)
