# Rust Cross-Market Arbitrage Bot

**Source:** [Trum3it/polymarket-arbitrage-bot](https://github.com/Trum3it/polymarket-arbitrage-bot)
**Recommendation:** NO (queue)

## What It Does

A Rust-based arbitrage bot for Polymarket that monitors ETH and BTC 15-minute price prediction markets using a **"market-neutral strategy"**.

### Strategy
1. Monitor ETH Up/Down and BTC Up/Down markets simultaneously
2. Look for opportunities where complementary tokens sum < $1.00
3. Example: ETH Up @ $0.47 + BTC Down @ $0.40 = $0.87 → $0.13 guaranteed profit
4. Execute simultaneous buy orders and wait for market resolution

### Architecture
- **api.rs** — Gamma API + CLOB API client
- **monitor.rs** — Continuous price fetcher
- **arbitrage.rs** — Opportunity detection
- **trader.rs** — Simulation/production execution

## Implementability: 2/5

- Rust has performance advantages for HFT-like scenarios
- Well-documented but narrow scope (only ETH/BTC 15-min)
- Better suited as a reference for a Python reimplementation

## Risks
- Limited to crypto markets only
- Competition in 15-min arb is intense
- Polymarket dynamic fees (2026) specifically target short-term crypto arb

## Next Steps
Use the market-neutral strategy concept in our Python cross-platform bot.
