# Trum3it Rust Arbitrage Bot

**Source:** https://github.com/Trum3it/polymarket-arbitrage-bot

## What It Does
A Rust-based arbitrage bot for Polymarket that monitors ETH and BTC price prediction markets (15-min and 1-hour) and executes trades using a "market-neutral strategy." The bot looks for opportunities where the sum of two complementary tokens (one from each market) is less than $1.00 — a guaranteed profit at resolution.

## Key Features
- **Market-neutral strategy:** Buys Up in ETH + Down in BTC (or vice versa) — uncorrelated directional bets
- **Dual market monitoring:** Both ETH and BTC 15-min price prediction markets
- **Configurable thresholds:** Minimum profit threshold, max position size, check interval
- **Simulation mode:** Test without real funds
- **Condition ID auto-discovery:** Can find market condition IDs automatically

## Why It Matters
The market-neutral approach (ETH Up + BTC Down) provides true risk-free profit — not dependent on market direction but on probability mispricing. This is a genuinely different approach from cross-platform arb or market making.

## Implementability: 3/5
**MEDIUM** — Rust implementation is clean but requires Rust toolchain. The combined probability arb (ETH Up + BTC Down) is the most innovative aspect. Need sufficient liquidity in both markets simultaneously for execution.

## Next Steps
1. Extend to SOL and XRP 15-min markets for more pair combinations
2. Add slippage analysis for the combined position entry
3. Automate condition ID discovery across all crypto markets
