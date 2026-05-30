# WSOL12 Polymarket-Kalshi BTC Arbitrage Scanner

**Source:** https://github.com/WSOL12/Polymarket-Kalshi-Arbitrage-Trading-Bot-BTC

## What It Does
Cross-platform scanner for Polymarket and Kalshi that detects pricing inefficiencies in hourly Bitcoin prediction markets. Continuously compares bid-ask data across both platforms and identifies spreads where the combined cost of taking opposite sides on the same outcome falls below the $1.00 settlement value.

## Architecture
- **Backend:** Python FastAPI — real-time price polling at 1-second intervals
- **Frontend:** Next.js + Tailwind CSS dashboard with live price feeds and cost breakdowns
- **Matching:** Automatic alignment of markets across platforms using strike price
- **Strategies:** Supports both directional combos (Down + Yes, Up + No)

## Key Features
- Continuous polling at 1-second intervals
- Auto-matching of equivalent markets across Polymarket and Kalshi
- Dashboards with live cost breakdown per position
- Both directional strategies supported

## Why It Matters
BTC hourly prediction markets are among the highest-volume venues on both Polymarket and Kalshi, making them ideal for cross-platform arbitrage. This bot provides a full-stack reference implementation with both backend logic and frontend visualization.

## Risks
- Requires API keys for both Polymarket (Gamma + CLOB) and Kalshi
- Execution risk: market orders may slip if liquidity thins
- Gas costs on Polygon can eat thin spreads
- Must monitor both platforms' uptime reliability

## Implementability: 4/5
Well-documented, full-stack application with clear setup instructions. The dual-platform API integration is the main complexity.

## Next Steps
1. Test with paper trading mode
2. Add support for ETH and other altcoin price markets
3. Extend to multiple strike windows simultaneously
