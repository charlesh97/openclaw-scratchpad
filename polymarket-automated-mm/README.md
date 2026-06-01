# Polymarket Automated Market Maker (Poly-Maker)

**Source:** https://github.com/terrytrl100/polymarket-automated-mm
**Recommendation:** YES — Directly applicable for liquidity provision strategy
**Implementability:** 5/5

---

## What It Does

A production-ready automated market making bot for Polymarket prediction markets. Maintains two-sided orders (buy + sell) simultaneously on selected markets, optimized for Polymarket's maker reward program. Originally created by @defiance_cr, this fork adds features and continued maintenance.

## Architecture

```
┌─────────────────────────────────────────┐
│           Poly-Maker Bot                 │
│                                           │
│  ┌──────────┐  ┌──────────────┐         │
│  │ poly_data │  │ poly_merger  │         │
│  │ (core MM) │  │ (positions)  │         │
│  └─────┬────┘  └──────┬───────┘         │
│        │              │                  │
│  ┌─────┴────┐  ┌──────┴───────┐         │
│  │ poly_stats│  │ poly_utils   │         │
│  │ (metrics) │  │ (utilities)  │         │
│  └──────────┘  └──────────────┘         │
│                                           │
│  ┌──────────────┐                         │
│  │ data_updater  │  (market scanner)     │
│  └──────────────┘                         │
│                                           │
│  Integration:                              │
│  - Google Sheets (config management)     │
│  - Polymarket WebSocket (real-time)      │
│  - CLOB API (order execution)            │
└─────────────────────────────────────────┘
```

## Key Features

| Feature | Details |
|---------|---------|
| **Strategy** | Two-sided market making (simultaneous bid + ask) |
| **Pricing** | Reward-optimized — uses Polymarket's maker reward formula |
| **Market Selection** | Data-driven: by profitability or daily reward |
| **Config** | Google Sheets — edit params without restart |
| **Position Mgmt** | Automated position merging, risk controls |
| **Order Churn** | Intelligent cancellation thresholds to minimize gas |
| **Reward Tracking** | Real-time estimated maker rewards |
| **WebSockets** | Real-time order book monitoring |

## How It Works

1. **Data Updater** — Fetches all available markets, calculates rewards and volatility metrics (runs continuously in background)
2. **Market Selection** — Ranked by profitability or minimum daily reward (e.g., `--min-reward 100 --max-markets 10`)
3. **Order Placement** — Simultaneous buy and sell orders at prices optimized for maker rewards
4. **Position Management** — Merges small positions, monitors risk, adjusts spreads
5. **Monitoring** — Real-time stats via poly_stats module

## Why This Matters

- **Directly deployable** — Python, no exotic dependencies
- **Revenue generation** — Polymarket's maker reward program pays for liquidity
- **Complements arb strategy** — Market making generates returns during periods with no arb opportunities
- **Proven track record** — Based on @defiance_cr's original, battle-tested on mainnet

## Risks

- Requires capital locked in positions (inventory risk)
- Impermanent loss on fast-moving markets
- Gas costs on Polygon (though minimized by churn thresholds)
- Maker rewards may change (Polymarket controls the program)

## Next Steps

1. Set up Google Sheets API credentials
2. Configure wallet with USDC on Polygon
3. Run `data_updater` to scan available markets
4. Start with `--min-reward 100` on 3-5 markets
5. Monitor for 1-2 weeks before scaling

## Integration with Arb Bot

The market making strategy pairs naturally with our arb bot:
- During calm markets: earn maker rewards as LP
- During volatile markets: arb bot captures dislocations
- Capital can be dynamically allocated between the two strategies
