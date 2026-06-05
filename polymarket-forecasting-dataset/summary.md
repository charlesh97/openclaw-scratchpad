# Summary: Unlocking the Forecasting Economy

**arXiv:2604.20421** — Published April 22, 2026

## Problem
Prediction market data is fragmented across heterogeneous on-chain and off-chain sources. No unified dataset existed for the full market lifecycle.

## Solution
First continuously maintained dataset suite integrating three layers:
1. Market metadata (770K+ records)
2. Fill-level trading records (943M+ fills)
3. Oracle-resolution events (2M+ events)

## Key Technical Contributions
- Identifier resolution across heterogeneous sources
- On-chain recovery for missing data
- Incremental update mechanism
- Reproducible collection pipeline

## Relevance to Arbitrage Research
- Largest backtesting corpus for Polymarket strategies ever published
- Allows validation of arb strategies across 5.5 years of market history
- Can test fee-aware strategies against pre- and post-fee-change periods
- Enables calibration research (NBA outcomes, CPI expectations)
