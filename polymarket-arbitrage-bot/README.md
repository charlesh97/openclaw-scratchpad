# Polymarket Arbitrage Bot — Rust, Market-Neutral

**Source:** https://github.com/Trum3it/polymarket-arbitrage-bot
**Recommendation:** MEDIUM — clean Rust implementation, limited to 15-min BTC/ETH

## What it does

A Rust-based arbitrage bot for Polymarket that monitors **ETH and BTC 15-minute** and **1-hour price prediction markets** using a "market-neutral strategy."

### Strategy
Buys complementary tokens across markets: ETH Up + BTC Down (or ETH Down + BTC Up). Since at least one token in each pair resolves to $1.00, if total acquisition cost < $1.00, profit is guaranteed.

### Architecture
- **API Client (api.rs)**: Gamma API + CLOB API communication
- **Market Monitor (monitor.rs)**: Continuous market data fetching
- **Arbitrage Detector (arbitrage.rs)**: Price analysis + opportunity detection
- **Trader (trader.rs)**: Trade execution (simulation or production)

## Implementability: 3/5
Rust is performant but less accessible. Limited to 2 market pairs. Good reference for implementing cross-condition arbitrage in Rust.

## Next Steps
1. Run in simulation mode to measure real opportunity frequency
2. Extend to additional crypto pairs (SOL, XRP)
3. Consider hybrid approach: Python for scanning, Rust for execution
