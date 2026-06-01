# Unlocking the Forecasting Economy — Polymarket Full-Lifecycle Dataset

**Source:** https://arxiv.org/abs/2604.20421
**Recommendation:** MEDIUM — Reference dataset for backtesting
**Implementability:** N/A (dataset, not algorithm)

---

## What It Is

The first continuously maintained dataset suite covering the full lifecycle of decentralized prediction markets on Polymarket. Built by Huaiyu Jia et al. (April 2026) — covers October 2020 to March 2026 with:
- **770K+** market records (metadata, creation, settlement)
- **943M+** fill-level trading records
- **~2M** oracle events (resolution, dispute, settlement)

## Dataset Architecture

The dataset integrates three canonical layers:

1. **Market Metadata Layer** — Creation, token registration, topic categorization, resolution criteria
2. **Fill-Level Trading Layer** — Every order fill: price, size, timestamp, maker/taker, condition ID
3. **Oracle-Resolution Layer** — Oracle votes, disputes, final outcomes, settlement transactions

Built with:
- Identifier resolution across off-chain and on-chain sources
- On-chain recovery for missing data
- Incremental updates (continuously maintained)

## Why This Matters

**For our arb bot project, this is the single most valuable dataset available:**

| Use Case | How it Helps |
|----------|-------------|
| **Backtesting arb strategies** | 943M fills across all market types |
| **Cross-platform analysis** | Can combine with Kalshi data for arb detection |
| **Feature engineering** | Rich fill-level data for ML features |
| **Market selection** | Identify most active markets by volume/fill count |
| **Timing analysis** | Find peak arbitrage windows by time of day/event type |
| **NBA outcome calibration** | Authors demonstrate NBA calibration use case |

## Key Findings from the Paper

- **NBA Outcome Calibration** — Shows how market prices can be calibrated against actual outcomes to measure prediction accuracy
- **CPI Expectation Reconstruction** — Demonstrates reconstructing macroeconomic expectation signals from prediction market prices
- **770K+ markets** tracked from creation through 943M fills to resolution

## Access

- Dataset: https://www.polymonitor.club/
- Code and pipeline: https://arxiv.org/abs/2604.20421
- HTML version: https://arxiv.org/html/2604.20421v1

## Next Steps

1. Download the dataset from polymonitor.club
2. Load fill-level data (943M rows) into local database for querying
3. Cross-reference with Kalshi data for cross-platform arb backtesting
4. Run our arb detection algorithms on the historical dataset
5. Profile most exploitable market categories and time windows
