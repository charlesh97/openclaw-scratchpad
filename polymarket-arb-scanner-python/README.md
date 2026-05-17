# Polymarket Arbitrage Scanner (Python)

**Source:** https://github.com/ImMike/polymarket-arbitrage  
**Recommendation:** MEDIUM

## What It Does

A Python-based arbitrage bot that watches 10,000+ markets across Polymarket and Kalshi for pricing inefficiencies. Detects three types of arbitrage: cross-platform price differences, bundle arbitrage (YES+NO ≠ $1.00), and market-making spread capture.

## Key Features

- **Cross-Platform Arbitrage** — Detects price differences between Polymarket and Kalshi
- **Bundle Arbitrage Detection** — YES + NO price not summing to ~$1.00
- **Market Making** — Captures spreads via competitive bid/ask orders
- Large-scale coverage (10,000+ markets)

## Why It Matters

Comprehensive coverage of multiple arbitrage types in a single Python codebase. The bundle arbitrage detection is especially useful for identifying simple market inefficiencies. Python makes it easy to customize and extend.

## Implementability: 4/5

Pure Python, numpy/pandas based. Easy to modify and understand. Well-structured for adding new detection types.

## Risks

- Must maintain high scan frequency for 10K+ markets
- Bundle arb opportunities are typically fleeting
- Cross-platform execution requires careful timing
