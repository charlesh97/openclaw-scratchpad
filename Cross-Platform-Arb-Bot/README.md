# Cross-Platform Arbitrage Bot (Polymarket + Kalshi)

**Source Repositories:**
- [ImMike/polymarket-arbitrage](https://github.com/ImMike/polymarket-arbitrage) — Python, scans 10,000+ markets
- [realfishsam/prediction-market-arbitrage-bot](https://github.com/realfishsam/prediction-market-arbitrage-bot) — Node.js, uses pmxt.dev
- [CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot](https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot) — Python + Next.js dashboard
- [WSOL12/Polymarket-Kalshi-Arbitrage-Trading-Bot-BTC](https://github.com/WSOL12/Polymarket-Kalshi-Arbitrage-Trading-Bot-BTC)

**Recommendation:** ✅ YES

## What It Does

Detects price differences between **Polymarket** and **Kalshi** for the same prediction market event, then executes simultaneous trades to lock in risk-free profit.

### Core Strategy: Synthetic Arbitrage

```
1. Find same event on both platforms
2. Buy YES on Platform A where it's cheaper
3. Buy NO on Platform B where it's cheaper
4. Total cost < $1.00 → guaranteed profit at resolution
```

**Example:**
- Polymarket: "Kevin Warsh YES" = **41¢**
- Kalshi: "Kevin Warsh NO" = **57¢**
- Total: **98¢**, Profit: **2¢ guaranteed**

## Why It Matters

- **Multiple implementations** (Python, Node.js, Rust) means the concept is battle-tested
- **Cross-platform spread** is structurally persistent — Kalshi and Polymarket have different user bases
- **$40M+ extracted** in single-platform arb alone (Probabilistic Forest paper)
- Cross-platform arb adds another layer of opportunity
- Polymarket's Feb 2026 removal of 500ms taker order delay makes execution faster

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│   Polymarket CLOB   │     │     Kalshi API       │
│   (Gamma API)       │     │                      │
└─────────┬───────────┘     └──────────┬───────────┘
          │                            │
          ▼                            ▼
    ┌──────────────────────────────────────┐
    │        Market Matching Engine        │
    │  (Fuzzy name matching: Jaccard +     │
    │   Levenshtein distance)              │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │      Arbitrage Opportunity Finder    │
    │                                      │
    │  Strategy 1: Poly YES + Kalshi NO    │
    │  Strategy 2: Poly NO + Kalshi YES    │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │       Execution Engine               │
    │  - Market orders (aggressive)        │
    │  - Simultaneous execution            │
    │  - Fee accounting                    │
    └──────────────────────────────────────┘
```

## Implementability: 4/5

- Well-documented with multiple reference implementations
- Live dashboards available (ImMike shows real-time scanning of 5,000+ markets)
- pmxt.dev provides a unified API across both platforms
- Main challenge: speed — opportunities may last only seconds
- Dry-run mode makes testing safe

## Risks

- **Execution risk**: Simultaneous fills not guaranteed; one side may slip
- **Liquidity risk**: Shallow order books on both platforms
- **Counterparty risk**: Platform settlement mechanisms differ
- **Gas costs**: Polygon fees, though low, add up at high frequency

## Next Steps

1. Clone ImMike/polymarket-arbitrage for the most complete Python implementation
2. Run in simulation mode on historical data
3. Add support for sports/event markets beyond crypto
4. Implement Telegram/email alerts for detected opportunities
5. Deploy with auto-execution after 2-week paper trading validation
