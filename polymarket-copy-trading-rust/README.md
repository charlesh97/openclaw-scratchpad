# Polymarket Copy Trading Bot (Rust)

**Source:** https://github.com/gamma-trade-lab/polymarket-copy-trading-bot
**Recommendation:** MEDIUM

## What It Does

A high-performance Rust implementation of a copy-trading bot for Polymarket. Focuses on real-time mirroring of top traders with sub-millisecond latency detection and execution.

### Key Features

- **Sub-millisecond execution latency** — 5-10x faster than Python/TypeScript equivalents
- **Multi-wallet tracking** — monitors 2-5 target wallets simultaneously
- **Copy trading**: mirrors top Polymarket traders with configurable multipliers
- **Low resource usage** — runs comfortably on small VPS instances
- **Order aggregation** — bundles trades for gas savings (25-45% reduction)
- **Persistent state** — SQLite/RocksDB with crash recovery

### Why Rust Matters for Polymarket

The February 2026 removal of the ~500ms taker delay on crypto markets fundamentally changed the landscape. Pure HFT-style arbitrage became much harder. Copy trading in Rust provides a measurable edge because:

- No GC pauses (unlike Python/Go/Java)
- Predictable sub-millisecond WebSocket processing
- 25-45% lower effective gas cost per mirrored trade
- Zero crashes in 50+ hours of live mainnet runs

### Architecture

```
Rust binary (single static binary)
├── WebSocket/HTTP polling (Tokio async)
├── Order aggregation engine (15-60s windows)
├── Tiered position sizing
├── In-memory + SQLite persistence
└── Config.toml for wallet/risk parameters
```

## Risks

- Copy trading is inherently dependent on target trader quality
- Slippage, gas spikes, and adverse selection remain real
- Not all copy-traded markets will be profitable
- Rust ecosystem — less accessible for Python-native teams

## Implementability: 4/5

Production-tested Rust binary with documented deployment flow. Single static binary means easy deployment. Requires Rust toolchain for compilation.

## Next Steps

1. Clone and configure target wallets
2. Start with dry-run / shadow mode
3. Begin with conservative multipliers (0.5x)
4. Analyze copy-trading alpha from top Polymarket traders
