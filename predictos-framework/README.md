# PredictOS — All-in-One Prediction Market OS

**Source:** https://github.com/PredictionXBT/PredictOS  
**License:** Open Source  
**Language:** TypeScript/Python  
**Added:** 2026-06-05

## What It Does

PredictOS is the most comprehensive open-source prediction market framework found to date. It's an "operating system" for prediction markets with:

- **Multi-market support** — Polymarket, Kalshi, Jupiter (Kalshi-based)
- **AI Super Intelligence** — Multi-agent AI system using xAI Grok, OpenAI GPT (up to GPT-5.2)
- **Dual modes** — Supervised (AI recommends, user approves) and Autonomous (AI executes within budget)
- **Cross-platform arbitrage** — Paste any market URL, auto-finds same market on other platforms, compares prices
- **Real-time data** via DFlow (Kalshi) and Dome (Polymarket) APIs
- **$PREDICT token ecosystem** — Staking, launchpad, governance

## Architecture

```
PredictOS/
├── terminal/         ← UI/dashboard
├── agents/           ← AI agent system
│   ├── predict-agents/     ← Individual AI agents (Grok, GPT)
│   ├── bookmaker-agent/    ← Consensus/judge agent
│   └── mapper-agent/       ← Order parameter mapping
├── connectors/       ← Platform integrations
│   ├── polymarket/
│   ├── kalshi/
│   └── jupiter/
└── docs/            ← Full documentation
```

The Super Intelligence pipeline:
1. **Predict Agents** — Multiple AI agents independently analyze markets (each with different tools/models)
2. **Bookmaker Agent** — Synthesizes all perspectives into a consensus recommendation
3. **Mapper Agent** — Translates analysis into platform-specific order parameters

## Why It Matters

- **Most complete framework** — Covers analysis, execution, arbitrage, copy trading
- **Multi-model AI** — Mix xAI Grok, OpenAI GPT-5.2, more coming
- **Self-hosted** — Your data never leaves your infrastructure; strategies stay private
- **Verifiable agents** — Store agent analysis on Irys blockchain for transparent audit trail
- **Active development** — Regular releases with clear roadmap

## Risks

- **Token dependency** — $PREDICT token required for full features (centralization risk)
- **Complex setup** — Multiple dependencies, Docker infrastructure
- **Model costs** — API calls to xAI/OpenAI add up
- **Early stage** — Still in active development, some features "coming soon"
- **Over-engineered** — May be more than needed for simple strategies

## Implementability: 4/5

Strong. Well-documented with full setup guide. Multi-market support means we get Polymarket + Kalshi arb built-in. The Super Intelligence system is novel but complex. Best for teams wanting a full-stack solution.

## Next Steps

1. Deploy PredictOS locally with Docker
2. Configure Polymarket and Kalshi API keys
3. Test Supervised mode with arbitrage strategies
4. Evaluate Autonomous mode with small budget ($1–$100)
