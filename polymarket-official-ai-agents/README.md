# Polymarket Official AI Agents Framework

**Source:** https://github.com/Polymarket/agents  
**License:** MIT  
**Language:** Python 3.9+  
**Stars:** Active (official Polymarket repo)  
**Added:** 2026-06-05

## What It Does

This is Polymarket's **official first-party framework** for building AI agents that trade autonomously on Polymarket. It provides:

- **Direct Polymarket API integration** — Market data retrieval, order execution, position management
- **AI agent utilities** — LLM tooling for prompt engineering, RAG (Retrieval-Augmented Generation)
- **Data connectors** — Chroma vector DB for news/RAG, Gamma API for market metadata
- **CLI interface** — Ready-to-use command-line trading interface
- **Modular architecture** — Community-extendable components

## Architecture

```
agents/
├── application/
│   └── trade.py          ← Main trading loop
├── connectors/
│   ├── chroma.py          ← Vector DB for news/data
│   ├── gamma.py           ← Polymarket Gamma API client
│   ├── polymarket.py      ← Core Polymarket API + DEX interaction
│   └── objects.py         ← Pydantic data models
└── scripts/
    ├── python/
    │   └── cli.py         ← User CLI interface
    └── bash/
        ├── build-docker.sh
        └── run-docker-dev.sh
```

## Why It Matters

This is **first-party infrastructure** from Polymarket themselves. Key implications:

1. **Trusted integration** — Built and maintained by the Polymarket team. No reverse-engineering needed.
2. **Official API access** — Direct access to CLOB, CTF exchange, and Gamma API
3. **LLM-native** — Designed from the ground up for AI agent integration (Langchain, Chroma)
4. **MIT License** — Fully open, no usage restrictions
5. **Low barrier** — `pip install && python cli.py` — minutes to first trade

This is the reference implementation that all Polymarket trading agents should be built on.

## Risks

- **US persons restricted** — Terms of Service prohibit US users from trading
- **LLM costs** — Relies on OpenAI API key; costs scale with usage
- **Wallet security** — Private key stored in .env
- **Early stage** — Community-maintained extensions may have varied quality
- **No built-in strategies** — Provides infrastructure, not trading logic

## Implementability: 5/5

**Excellent.** MIT-licensed, official Polymarket repo, Python, well-documented, Docker support. Clone, `pip install -r requirements.txt`, set API keys, run CLI. Perfect foundation for building any Polymarket trading agent.

## Next Steps

1. Clone and test CLI with paper trading
2. Build strategy modules on top of the connector layer
3. Integrate with our existing arbitrage detection pipeline
4. Contribute back — Polymarket explicitly accepts PRs
