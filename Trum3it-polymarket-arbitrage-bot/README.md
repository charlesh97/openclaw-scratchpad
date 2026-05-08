# Trum3it/polymarket-arbitrage-bot (Rust)

**Source:** https://github.com/Trum3it/polymarket-arbitrage-bot

## What It Does
A **Rust-based** arbitrage bot for Polymarket's 15-minute ETH and BTC price prediction markets. Monitors ETH-up/down and BTC-up/down tokens simultaneously. When ETH-UP + BTC-DOWN (or vice versa) costs less than $1.00 in combined ask prices, it buys both legs — guaranteed ~$1 payout at resolution regardless of which tokens win.

The market-neutral "straddle" style: you're betting on the relative relationship between two correlated assets, not their absolute direction.

## Architecture (Rust)
- `api.rs` — Polymarket Gamma + CLOB API client
- `monitor.rs` — continuous polling of ETH and BTC 15-min markets
- `arbitrage.rs` — pairwise price combination detection
- `trader.rs` — execution in simulation or production mode
- Config via `config.json` (created on first run)

## Key Config
- `min_profit_threshold: 0.01` — minimum profit in dollars per trade
- `max_position_size: 100.0` — max per trade
- `check_interval_ms: 1000` — poll every second

## Implementability: 2/5
- Requires **Rust toolchain** — steeper than Python for most algo traders
- Active development; good test coverage with `--simulation` mode
- Targets a narrow niche: only 15-min BTC/ETH price markets (not general prediction markets)
- Auto-discovers condition IDs, which is helpful but the scope is limited
- Pair trading on correlated assets is a real edge, but Rust barrier is real

## Risks
- Same shallow order book issue: max executable size is small
- Only works for the 15-min BTC/ETH window markets (not general arb)
- Requires Polymarket API key for production mode
- Straddle strategy requires both legs to fill simultaneously — partial fills are a risk

## Next Steps
1. Study the Rust source to understand the condition-discovery logic
2. Consider whether the paired straddle approach can be generalized to other asset pairs
3. For Python-first traders, the ImMike bot covers more market types with similar logic
