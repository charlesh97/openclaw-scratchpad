# Polymarket Arbitrage Bot (ImMike)

**Source:** https://github.com/ImMike/polymarket-arbitrage
**Recommendation:** YES ✅
**Implementability:** 4/5

## What It Does

A comprehensive Python-based arbitrage bot that scans **10,000+ Polymarket markets** in real-time and detects three types of opportunities:

1. **Bundle Arbitrage** — YES + NO prices not summing to $1.00 (intra-market)
2. **Cross-Platform Arbitrage** — Price differences between Polymarket and Kalshi for the same prediction
3. **Market Making** — Captures spreads by placing competitive bid/ask orders

Complete with a **live web dashboard** (FastAPI), risk management system, dual data modes (real/simulation), and comprehensive logging.

## Architecture

```
├── main.py                    # Entry point
├── polymarket_client/         # Polymarket API (CLOB + WebSocket)
├── kalshi_client/             # Kalshi REST API integration
├── core/
│   ├── arb_engine.py          # Single-platform opportunity detection
│   ├── cross_platform_arb.py  # Cross-platform arb (Polymarket ↔ Kalshi)
│   ├── execution.py           # Order management
│   ├── risk_manager.py        # Position limits, loss limits, kill switch
│   └── portfolio.py           # Position & PnL tracking
├── dashboard/                 # FastAPI + real-time web UI
└── utils/                     # Config, logging, backtesting
```

## Why It Matters

- **Production-ready** codebase with proper test suite
- **Live dashboard** provides real-time visibility — not a black box
- **Simulation mode** for safe strategy testing before committing funds
- **Cross-platform** (Polymarket + Kalshi) — largest opportunity surface area
- **Market Matching AI** — uses text similarity to auto-match predictions across platforms
- Active, well-documented, MIT licensed

## Risks

- Real markets are highly efficient — arbitrage opportunities are rare in practice
- Requires both Polymarket and Kalshi API credentials
- Gas costs and taker fees can eat thin margins
- The paper shows 99.6% win rate in simulation — real-world will be lower

## Next Steps

1. Clone repo, install dependencies (`pip install -r requirements.txt`)
2. Configure `config.yaml` with API keys
3. Run simulation mode to observe detection patterns
4. Deploy with minimal bet size to validate real-world profitability
5. Extend with additional exchange integrations (Robinhood? PredictIt?)

## Implementability Score: 4/5

Ready to run. Only delta is real-world arb frequency vs simulation. Strong foundation to build on.
