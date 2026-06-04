# Polymarket-Kalshi Cross-Platform Arbitrage Bot

**Source:** https://github.com/ImMike/polymarket-arbitrage  
**Recommendation:** YES ✅  
**Language:** Python 3.10+  

## What It Does

A comprehensive arbitrage bot that monitors 10,000+ prediction markets across Polymarket and Kalshi simultaneously. It detects three types of arbitrage:
1. **Cross-Platform Arbitrage** — price differences for the same event between Polymarket and Kalshi
2. **Bundle Arbitrage** — YES + NO prices not summing to ~$1.00 on the same market
3. **Market Making** — captures spreads with competitive bid/ask orders

## Key Features

- **Cross-Platform Detection** — automatically matches similar predictions across platforms using text similarity AI
- **Real-time Dashboard** — web UI showing live opportunities and bot activity
- **Dual Data Modes** — simulation (for testing) and live (real Polymarket data)
- **Fee Accounting** — realistic edge calculations including gas costs and platform fees
- **Risk Management** — position limits, daily loss limits, kill switch
- **Market Matching AI** — NLP-based matching of similar predictions across platforms

## Architecture

```
config.yaml → main.py → Market Scanner (10k+ markets)
                         ├── Cross-Platform Arb Detector
                         ├── Bundle Arb Detector  
                         ├── Market Maker
                         └── Risk Manager
Streamlit Dashboard ← Results
```

## Why It Matters

- Scans **5,000+ live Polymarket markets** in real-time
- Cross-platform arb is the most accessible alpha source for retail
- Simulation mode achieved 99.6% win rate with $573 profit in testing
- Complete fee accounting including gas costs
- Active, well-maintained codebase

## Risks

- Real markets are highly efficient — arb opportunities are rare in production
- Cross-platform arb requires accounts on both Polymarket AND Kalshi
- Gas fees on Polygon can eat small arb spreads
- Polymarket's 2026 dynamic fee model reduces short-term arb viability
- Text similarity matching can mis-match different prediction questions

## Implementability: 5/5

Best-in-class documentation, clear config, one-command setup, both simulation and live modes. Perfect starting point for prediction arb.

## Next Steps

1. Run in simulation mode to validate strategy parameters
2. Configure Kalshi API credentials for cross-platform scanning
3. Tune minimum edge threshold for current fee environment
4. Monitor live for 1-2 weeks before enabling auto-trading
