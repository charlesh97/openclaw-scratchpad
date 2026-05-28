# Gamma Trade Lab — Polymarket Copy Trading Bot (Rust)

**Source:** [github.com/gamma-trade-lab/polymarket-copy-trading-bot](https://github.com/gamma-trade-lab/polymarket-copy-trading-bot)

## What It Does

A **high-performance Rust implementation** of a copy trading bot for Polymarket that mirrors top trader positions in real-time. Optimized for the **post-February 2026 latency improvement** when Polymarket removed the ~500ms artificial delay on taker orders for crypto markets.

## Key Features

- **Real-time wallet mirroring** — detects top trader activity and mirrors positions
- **Low-latency Rust core** — sub-100ms detection-to-execution pipeline
- **Post-delay-removal optimized** — takes advantage of Polymarket's 2026 latency infrastructure changes
- **Copy trading focus** — follows proven profitable traders' moves
- **Portfolio tracking** — monitors multiple target wallets simultaneously

## Why It Matters

Polymarket's removal of the 500ms taker order delay in February 2026 fundamentally changed the latency landscape. This bot specifically targets the post-change environment where:
- Copy trading is more viable (faster execution mirrors faster detection)
- Latency-arbitrage bots now compete differently
- The ~500ms advantage window has collapsed, favoring efficient execution over raw speed

## Risks

- Copy trading inherits all risks of target traders' strategies
- Polymarket may re-introduce latency delays
- Copy trading is detectable by target traders (they can front-run)
- Rust implementation requires Rust toolchain expertise to modify
- No guarantee top traders will continue to be profitable

## Implementability: 4/5

Production-quality Rust codebase with clear copy trading logic. Easy to deploy and customize. The latency optimization is particularly relevant post-Feb 2026.

## Next Steps

1. Set up target trader identification (profitable wallet discovery)
2. Benchmark execution latency vs. other copy trading implementations
3. Test with small capital in paper trading mode
4. Monitor Polymarket latency infrastructure for changes
