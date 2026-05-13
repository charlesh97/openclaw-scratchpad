# Polymarket Copy Trading Bot (Rust)

**Source:** https://github.com/gamma-trade-lab/polymarket-copy-trading-bot

## What it does

An enhanced Rust-based copy trading bot for Polymarket that mirrors top traders' positions in real-time. Focuses on low-latency detection and execution. Benefits from Polymarket's early 2026 removal of the 500ms artificial delay on taker orders for crypto markets.

## Key features

- **Top trader tracking:** Identifies profitable wallets via on-chain analysis
- **Real-time mirroring:** Sub-second detection of whale trades
- **Risk management:** Per-trader limits, drawdown protection
- **Rust performance:** Sub-millisecond order processing

## Why it matters

Copy trading is the lowest-effort strategy — piggyback on traders who've already proven their edge. Rust implementation means it can execute faster than Python-based competitors. The removal of the 500ms delay means copy traders can now react in real-time.

## Implementability: 3/5

Requires Rust toolchain and understanding of Polymarket's on-chain data. More complex to customize than Python alternatives but faster.

## Risks

- Lagging the copied trader by even 100ms can erode profits
- Whale wallets may manipulate by taking opposite positions
- Requires reliable RPC node for on-chain data

## Next Steps

1. Identify top 50 profitable Polymarket wallets
2. Set up Rust development environment
3. Configure copy trading parameters (min trade size, max allocation)
4. Deploy with RPC redundancy
