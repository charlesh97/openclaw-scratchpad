# Polymarket Official AI Agents Framework

**Source:** https://github.com/Polymarket/agents
**Recommendation:** YES — **TOP 2 for today's email**

## What It Does

The official Polymarket AI Agents framework provides a developer toolkit and set of utilities for building AI agents that trade autonomously on Polymarket. It's the **official, first-party** framework from Polymarket itself — not a third-party project.

### Key Features

- **Polymarket API Integration**: First-class connectors to Polymarket's Gamma API, CLOB API, and data feeds
- **LLM-Powered Trading**: Built-in utilities for connecting AI models (OpenAI, Claude) to market analysis and trade execution
- **RAG (Retrieval-Augmented Generation)**: ChromaDB-based vector storage for news, market data, and custom datasets
- **CLI Interface**: `python cli.py` commands for get-all-markets, trade, analyze, and more
- **Modular Architecture**: Separate connectors for market data (Gamma), order execution (Polymarket), and data objects (Pydantic models)
- **MIT Licensed**: Free and open source

### Architecture

```
polymarket-agents/
├── agents/application/trade.py    # Main trading application
├── scripts/python/cli.py          # CLI interface
├── connectors/
│   ├── chroma.py                  # Vector DB for RAG
│   ├── gamma.py                   # Market metadata client
│   └── polymarket.py              # Order execution + market data
├── objects.py                     # Pydantic data models
└── requirements.txt               # Python dependencies
```

## Why It Matters

This is not just another bot — it's the **official reference implementation** from the Polymarket team for AI-powered trading. Any serious prediction market automation effort should start by understanding this framework. It sets the standard for:

- How to authenticate and sign orders on the CLOB
- How to fetch and parse market metadata
- How to structure AI agent interactions with market data

## Risks

- Requires Python 3.9 (somewhat dated)
- Documentation is developer-focused with limited usage examples
- The CLI interface is functional but not a polished trading UI
- MIT license means no warranty for financial trading use

## Implementability: 5/5

Production-ready official framework with complete CLI + Python API. Clone, configure `.env`, and start trading.

## Next Steps

1. Clone and run the CLI to explore markets
2. Build custom AI agents using the connector architecture
3. Integrate with our existing strategy modules
4. Extend with custom RAG data sources for domain-specific insights
