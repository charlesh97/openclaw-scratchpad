# Unlocking the Forecasting Economy — Polymarket Dataset Suite

**Source:** https://arxiv.org/abs/2604.20421  
**Authors:** Huaiyu Jia et al.  
**Type:** Research paper + dataset  
**Date:** April 2026  
**Added:** 2026-06-05

## What It Provides

The first **continuously maintained dataset suite** for the full lifecycle of decentralized prediction markets, built on Polymarket. Covers **October 2020 to March 2026**:

- **770,000+ market records** — Metadata, categories, creation details
- **943 million fill records** — Every trade executed
- **2 million oracle events** — Resolution outcomes, disputes, settlements
- **Unified relational schema** — Three canonical layers integrated: metadata, fills, oracle events

## Architecture

```
Dataset Schema:
├── market_metadata    ← Market creation, conditions, tokens
├── fill_records       ← Every trade: price, size, side, timestamp
├── oracle_events      ← Resolution: outcome, dispute, finalization
└── identifiers        ← Cross-source ID resolution (on-chain + off-chain)
```

## Why It Matters

This is the **largest publicly available prediction market dataset ever released**. Strategic value:

1. **Backtesting at scale** — 943M fills over 5.5 years of Polymarket history
2. **Cross-source integration** — On-chain trade records joined with off-chain metadata
3. **Lifecycle completeness** — From market creation → trading → oracle resolution
4. **Reproducible pipeline** — Collection code is extensible and documented
5. **Two case studies included** — NBA outcome calibration + CPI expectation reconstruction

## Key Findings from the Paper

- PolyMarket dataset spanning 5.5 years shows massive scale growth
- NBA outcome markets show systematic calibration bias
- CPI expectation reconstruction reveals prediction markets as viable economic indicators

## Risks

- **Size** — 943M fills is enormous; requires serious infra to process
- **Static snapshot** — Dataset is a snapshot, not live
- **Data quality** — Cross-source reconciliation is imperfect
- **No strategy code** — Dataset + analysis, not trading logic

## Implementability: 3/5 (as research resource)

Essential reference dataset. Best-in-class for backtesting and calibration research. Not directly executable but invaluable for strategy validation.

## Next Steps

1. Download the dataset from [polymonitor.club](https://www.polymonitor.club/)
2. Backtest our arb detection algorithms against historical data
3. Validate fee-aware strategy performance over the fee-change period
4. Use case studies as calibration benchmarks
