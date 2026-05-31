# MrFadiAi Polymarket Bot (4-Strategy)

**Source:** https://github.com/MrFadiAi/Polymarket-bot

## What it does
A Polymarket trading bot with 4 strategies, dashboard, multi-strategy support, and auto-rotation (v3.1 — January 2026). Features enhanced risk management, smart money tracking, and dynamic position sizing.

## Core Features
- **4 Strategies** — Includes smart money following, trend, mean reversion, and momentum
- **Dashboard** — Web-based UI for monitoring and control
- **Auto-rotation** — Automatically switches between strategies based on market conditions
- **Smart Money Tracking** — Follows large/experienced traders
- **Dynamic Sizing** — Adjusts position size based on volatility and account equity

## Why it matters
Auto-rotation between strategies is a valuable concept. Markets change regime — what works for political events may not work for sports. An arb bot should also adapt its approach.

## Implementability: 4/5
TypeScript + React. Well-maintained with active releases. The smart money tracking feature is particularly interesting to integrate.

## Next Steps
1. Study the auto-rotation logic
2. Integrate smart money tracking signals into arb detection
