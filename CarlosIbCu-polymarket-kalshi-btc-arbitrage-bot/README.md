# CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot

**Source:** https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot

## What It Does
A **real-time arbitrage bot** specifically for the **Bitcoin 1-Hour Price market** between Polymarket and Kalshi. Monitors the BTC-hourly prediction market on both platforms simultaneously — when the combined cost of opposing legs (Poly Down + Kalshi Yes, or Poly Up + Kalshi No) is less than $1.00, a risk-free arb exists.

Unlike the ImMike bot which scans 5,000+ markets, this is laser-focused on a single, high-volume market (BTC hourly) where liquidity is deepest, making executable sizes somewhat larger than typical Polymarket arb.

## Architecture
- **Backend:** Python + FastAPI + Uvicorn + Requests
- **Frontend:** Next.js + shadcn/ui + TailwindCSS + Lucide React
- `backend/api.py` — serves real-time arb data on port 8000
- `frontend/` — live dashboard at localhost:3000

## Key Config
- Bot fetches live prices every **second**
- Smart event matching normalizes prices to 0.00-1.00 probability format
- Checks multiple leg combinations per tick

## Implementability: 4/5
- Very clean, focused scope — one market type, deeply optimized
- Great for understanding the specific Polymarket-Kalshi BTC arb thesis
- Includes a detailed [Arbitrage Thesis](https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot/blob/main/thesis.md) document
- FastAPI backend is easy to extend or embed into larger systems
- The single-market focus makes backtesting tractable

## Risks
- Extremely narrow scope (only BTC 1-hour) — only applicable when that market is active
- Requires both Polymarket API access AND Kalshi account (US-based, KYC'd)
- BTC hourly markets resolve quickly (within the hour) — timing risk is real
- Requires both legs to fill at the same price levels — partial fill risk
- Shallow order books still cap executable size

## Next Steps
1. Read the `thesis.md` — it's a clean theoretical grounding for the arb strategy
2. Run the backend API in isolation to get a feel for the data stream
3. Extend to other market pairs once the core logic is validated
