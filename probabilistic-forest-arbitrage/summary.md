# Arbitrage in Prediction Markets — 1-Page TL;DR

## The Question
How much arbitrage exists on Polymarket, and what types?

## Method
Analyzed 86M on-chain transactions across 17,218 conditions (Apr 2024 — Apr 2025). Used integer programming to detect two types of arbitrage.

## Key Results
- **$40M+ extracted** by arbitrageurs in one year
- **7,051 conditions** had single-market (rebalancing) arb
- **Combinatorial arb** exists across related markets but is harder to capture
- Median market price deviates ~40% from fair value ($0.60 out of $1.00)

## Takeaway
Arbitrage is real, persistent, and quantifiable on Polymarket. Single-market arb is low-hanging fruit; combinatorial arb offers higher margins per trade but requires condition-dependency mapping.
