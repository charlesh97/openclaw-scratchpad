# Polymarket-Kalshi BTC Arbitrage Bot (CarlosIbCu)

**Source:** https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot

## What It Does
Real-time arbitrage detection for the Bitcoin 1-Hour Price market between Polymarket and Kalshi. Full-stack: Python/FastAPI backend + Next.js dashboard with Docker Compose deployment.

## Key Features
- **Real-time monitoring:** Fetches live prices every second
- **Smart matching:** Auto-matches Polymarket events with corresponding Kalshi markets
- **Multi-strategy detection:** Poly Down + Kalshi Yes, Poly Up + Kalshi No
- **Beautiful dashboard:** Next.js with shadcn/ui, Tailwind CSS
- **One-command deploy:** `make build` — Docker Compose handles everything
- **Detailed arbitrage thesis:** Includes a `thesis.md` explaining the math

## Implementability: 5/5
**YES** — full-stack, production-ready with Docker. One-command deployment. Clear arbitrage thesis document. BTC 1H is the deepest cross-platform arb market. Dashboard makes monitoring easy.

## Next Steps
1. Deploy with Docker Compose
2. Add Telegram/Slack notifications for detected opportunities
3. Extend from BTC to ETH 1H markets
4. Add automated execution (currently detection-only)
