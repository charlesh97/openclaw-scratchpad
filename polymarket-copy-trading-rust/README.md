# Polymarket Copy Trading Bot (Rust) — gamma-trade-lab

**Source:** https://github.com/gamma-trade-lab/polymarket-copy-trading-bot

## What It Does
High-performance Rust copy-trading bot for Polymarket. Real-time mirroring of top traders with low-latency detection and execution. Leverages Polymarket's removal of the ~500ms taker delay (Feb 2026) to focus on copy-trading alpha instead of latency arb.

## Why It Matters
- **Rust advantage:** Sub-millisecond execution, no GC pauses, 99.9% WebSocket uptime
- **25–45% lower gas costs** vs Python/TS implementations (in-memory aggregation)
- **Multi-wallet tracking** (2–5 targets) with per-wallet risk params
- **Tiered position sizing** with confidence multipliers
- **Dry-run/shadow mode** for risk-free testing

## Key Insight
After Feb 2026's delay removal, pure HFT/latency arb became much less profitable. Copy-trading captures directional alpha from proven traders instead of fighting for microseconds.

## Implementability: 3/5
**MEDIUM** — requires Rust toolchain knowledge. Copy-trading strategy quality depends entirely on trader selection (need reliable signals). Performance advantages are real but diminishing as more Rust bots enter the space.

## Next Steps
1. Build a Polymarket whale tracker to identify consistently profitable wallets
2. Port the copy-trading strategy to Python for rapid prototyping
3. Run shadow mode on 2-3 target wallets for 2 weeks
