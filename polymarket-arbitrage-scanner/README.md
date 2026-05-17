# Polymarket Arbitrage Scanner (ImMike)

**Source:** https://github.com/ImMike/polymarket-arbitrage

## What It Does
Cross-platform arbitrage detection bot for Polymarket and Kalshi. Watches 10,000+ markets looking for inefficient markets on and between platforms. Features bundle arbitrage detection (YES+NO ≠ $1), market making, and a live dashboard.

## Key Features
- **Cross-platform arb:** Detects price differences between Polymarket and Kalshi
- **Bundle arb detection:** Identifies when YES + NO ≠ $1.00
- **Market matching AI:** Auto-matches similar predictions across platforms using text similarity
- **Live dashboard:** Real-time web UI showing opportunities and bot activity
- **Dual data modes:** Real market data or simulation mode
- **Fee accounting:** Realistic edge calculations including gas fees
- **99.6% win rate in simulation** with $573 profit (simulated)

## Implementability: 4/5
**MEDIUM** — full Python project with configuration, dashboard, and simulation mode. The market matching AI is particularly interesting for cross-platform arb. Real markets are highly efficient though — arb opportunities are rare in production.

## Next Steps
1. Run scanner in simulation mode to understand the arb landscape
2. Focus on niche markets (less liquid = more inefficiency)
3. Add notification system (Telegram/Slack) for real arb events
