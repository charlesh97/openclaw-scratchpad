# Polymarket Copy Trading Bot — Rust-Based Real-Time Wallet Mirroring

**Source:** https://github.com/gamma-trade-lab/polymarket-copy-trading-bot  
**Status:** Active (Feb 2026)  
**Language:** Rust  
**Recommendation:** MEDIUM — Strong execution reference, Rust advantage

## What It Does

A high-performance copy trading bot for Polymarket written in Rust. It focuses on real-time mirroring of top traders with emphasis on low-latency detection and execution. Notably, it exploits the removal of Polymarket's ~500ms artificial delay on taker orders (Feb 2026) to achieve competitive execution speed.

## Why It Matters

- **Rust implementation** offers a reference for high-performance execution (our current bot is Python)
- Exploits a specific platform change (delay removal) that expanded the latency arb window
- Wallet tracking + intelligent filtering (not just blind copying)
- Real-time order book integration

## Architecture

```
polymarket-copy-trading-bot/
├── tracker/     — Wallet monitoring and trade detection
├── filter/      — Quality filters for wallets to copy
├── executor/    — Low-latency order placement
└── metrics/    — Performance tracking and PnL attribution
```

## Implementability: 3/5

- Rust codebase — significant rewrite for our Python stack, or valuable if we port core components
- Copy trading introduces latency dependency between detection and execution
- Wallet quality filtering is the critical differentiator (good filters = alpha, bad = losses)

## Risks

- General copy-trading risk: followed wallets may be engaging in wash trading or have undisclosed strategies
- Platform changes can invalidate filter assumptions overnight
- Rust maintenance burden if team lacks Rust expertise

## Next Steps

1. Evaluate whether Rust performance gains justify a hybrid architecture (Rust executor + Python analysis)
2. Study wallet filtering methodology — could improve our Smart Money strategy
3. Backtest copy trading on historical wallet-level data before deploying capital
