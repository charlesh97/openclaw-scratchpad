# ImMike/polymarket-arbitrage

**Source:** https://github.com/ImMike/polymarket-arbitrage

## What It Does
A Python-based Polymarket + Kalshi cross-platform arbitrage bot that watches 5,000+ markets in real-time. Detects three types of opportunities:
1. **Cross-Platform Arb** — same prediction priced differently on Polymarket vs Kalshi (e.g., YES $0.52 on Poly vs $0.58 on Kalshi → buy Poly, sell Kalshi for 6% edge)
2. **Bundle Arb** — YES + NO prices don't sum to ~$1.00 on the same platform → buy both, guaranteed $1 payout
3. **Market Making** — places bid/ask inside wide spreads (≥5¢) to capture the spread

## Architecture
- `polymarket_client/` — Gamma REST API + WebSocket for live Polymarket data
- `kalshi_client/` — Kalshi REST API for cross-platform price data
- `core/arb_engine.py` — single-platform bundle detection (YES + NO < $1.00)
- `core/cross_platform_arb.py` — cross-platform matching + arbitrage detection
- `core/execution.py` — order management
- `core/risk_manager.py` — position limits, daily loss stop, kill switch
- `dashboard/server.py` — FastAPI live dashboard (localhost:8000)
- Uses text-similarity AI to auto-match "same" prediction across platforms

## Key Config
- `trading_mode: dry_run| live` — always start dry run
- `min_edge: 0.01` — 1% minimum profit after fees required
- `mm_enabled: true` — market making on/off
- `max_position_per_market: 15`, `max_global_exposure: 50` (start small)

## Implementability: 4/5
- Well-structured, readable Python with modular components
- Production-grade risk management (kill switch, daily loss limits)
- Full simulation mode for backtesting before live trading
- No Rust/HTTP/WS expertise required — Python + basic API knowledge enough
- Cross-platform logic is the most operationally complex part (two API integrations)

## Risks
- Real markets are highly efficient — arb opportunities are rare and fleeting (often <4 seconds)
- Requires **Polymarket API key** (private key) and **Kalshi access** (KYC'd US account)
- Kalshi trading requires US residency + identity verification
- Execution latency matters: HFT bots already capture most visible arb within milliseconds
- Small executable sizes due to shallow order books (confirmed by NBA paper: median 14.8 shares)

## Next Steps
1. Clone and run in `simulation` mode first, verify dashboard works
2. Get Polymarket API credentials; test in `dry_run` with `data_mode: real`
3. Study the cross-platform matching logic — the text-similarity approach is novel and auditable
4. Paper-trade for 1 week before sizing up
