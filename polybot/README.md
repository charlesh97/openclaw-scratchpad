# Polybot — Strategy Reverse-Engineering Toolkit

**Source:** https://github.com/ent0n29/polybot
**Recommendation:** MEDIUM
**Implementability:** 3/5
**Last updated:** ~2026

## What it does

Open-source Polymarket trading infrastructure and strategy reverse-engineering toolkit. Focuses on understanding what profitable traders are doing by analyzing on-chain data and order flow.

### Key features:
- Market data crawler for backtesting
- Trader intelligence (PSI indices)
- Fund mirroring capabilities
- API and UI layers

## Why it matters

The reverse-engineering approach is unique — instead of building strategies from first principles, it extracts signals from successful traders' patterns. The foundation for AWARE (trader intelligence product layer).

## Risks
- Alpha decays as more people mirror same traders
- Requires significant infrastructure (Rust + databases)
- Strategy quality depends on trader selection

## Next Steps
1. Crawl historical data for backtest corpus
2. Identify top-performing wallets to mirror
3. Build trader scoring model
