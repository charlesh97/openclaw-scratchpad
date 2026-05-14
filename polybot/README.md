# Polybot — Polymarket Trading Infrastructure & Strategy Reverse-engineering Toolkit

**Source:** https://github.com/ent0n29/polybot  
**Status:** Active  
**Language:** Python  
**Recommendation:** MEDIUM — Strong research value, infrastructure layer

## What It Does

Polybot is an open-source Polymarket trading infrastructure and strategy reverse-engineering toolkit. It provides the execution and market-data foundation for AWARE (the next product layer: trader intelligence, PSI indices, fund mirroring, API/UI). Think of it as the "data layer" for understanding how Polymarket strategies actually work by analyzing on-chain activity.

## Why It Matters

- **Reverse-engineering approach** is unique — instead of implementing strategies, it analyzes what others are doing and exposes those patterns
- Includes PSI (Polymarket Sentiment Index) calculation methodology
- Fund mirroring capability can automatically replicate successful strategies
- Provides a complete execution foundation (order placement, position tracking, PnL calculation)

## Architecture

```
polybot/
├── data/             — Market data ingestion (WebSocket + REST)
├── analysis/         — Strategy detection from trade patterns
├── mirror/           — Fund/copy trading engine
├── psi/              — Polymarket Sentiment Index calculation
└── exec/             — CLOB execution layer
```

## Implementability: 3/5

- Data analysis layer is strong; execution layer needs hardening for production
- AWARE layer is still in development
- Good reference for building market intelligence pipelines

## Risks

- Reverse-engineering detected strategies may lag behind actual profitable strategies
- Fund mirroring inherits all risks of followed wallets (wash trading, exit scams)
- PSI index construction methodology needs independent validation

## Next Steps

1. Evaluate PSI index methodology — could feed into our own signal generation
2. Backtest whether wallet-mirroring strategies outperform buy-and-hold on matched events
3. Extract data ingestion pipeline as a reusable component for our bot
