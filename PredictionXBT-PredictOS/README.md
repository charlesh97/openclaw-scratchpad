# PredictionXBT/PredictOS

**Source:** https://github.com/PredictionXBT/PredictOS

## What It Does
An **all-in-one open-source framework** for prediction market trading. Think of it as an operating system for building custom AI agents and trading bots across Polymarket, Kalshi, and Jupiter. The most relevant module for arb work is **Arbitrage Intelligence** — paste any market URL and it auto-searches for the equivalent on the other platform, compares prices, and calculates actionable profit.

Also includes: Super Intelligence (multi-agent AI market analysis), Betting Bots (Polymarket 15-min up/down ladder mode), Wallet Tracking, and Verifiable Agents (Irys blockchain audit trail).

## Architecture
- **Frontend:** Next.js 14 + TailwindCSS + shadcn/ui
- **Backend:** Supabase Edge Functions (Deno runtime)
- **Data providers:** Dome API (Polymarket), DFlow API (Kalshi/Jupiter)
- **AI:** xAI Grok + OpenAI GPT models via unified AI layer
- **Key file:** `supabase/functions/arbitrage-finder/` — cross-platform arbitrage endpoint

## Key Features for Arbitrage
- Text-similarity matching across platforms (paste Polymarket URL → finds Kalshi equivalent)
- Real-time price comparison with profit calculation
- Supervised (human approves) and Autonomous (auto-executes within budget) modes
- `$PREDICT` token staking for premium features (not required for basic arb detection)

## Implementability: 4/5
- Well-documented with separate setup guides per feature
- Docker + Supabase local dev environment works out of the box
- Multi-agent pipeline is sophisticated — Bookmaker Agent synthesizes multiple AI perspectives
- Good for building proprietary signal layers on top of the open framework
- MIT license, active community

## Risks
- Requires Supabase CLI + Docker for local dev — slightly complex setup
- DFlow API requires separate API key (contact DFlow directly)
- Autonomous mode requires Polymarket wallet private key + proxy wallet address
- The `$PREDICT` token ecosystem may be a distraction from the core arbitrage logic
- AI agent orchestration adds latency — not suitable for HFT-style millisecond arb

## Next Steps
1. Run the arbitrage-finder endpoint locally — it gives the cleanest cross-platform comparison
2. Compare its market-matching accuracy vs ImMike's text-similarity approach
3. Use the supervised mode to validate signals before considering autonomous execution
