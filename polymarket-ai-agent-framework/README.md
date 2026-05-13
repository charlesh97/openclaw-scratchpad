# Polymarket AI Agent Framework

**Source:** https://github.com/Polymarket/agents

## What it does

The official Polymarket AI Agent Framework for building autonomous trading agents. Provides a complete SDK for connecting LLM-based agents to Polymarket's CLOB (Central Limit Order Book) and Data APIs. Agents can retrieve news, query local data, send data/prompts to LLMs, and execute trades autonomously.

## Architecture

- **Agent Orchestrator:** Coordinates between LLM reasoning, data fetching, and trade execution
- **Polymarket API Integration:** Direct access to CLOB for order placement, Data API for market info
- **Tool-based interface:** Each capability (trade, search, news) is a tool the LLM can call
- **Plugin system:** Extensible for custom strategies

## Why it matters

This is the **official** bot-building framework from Polymarket itself. It validates the legitimacy of automated trading on the platform and provides the most up-to-date API integrations. Any serious bot builder should start here.

## Risks

- LLM latency may miss fast arbitrage windows
- API rate limits on free tier
- Requires managing private keys securely

## Implementability: 5/5

Ready-to-use framework with documentation. Just need an API key and a strategy.

## Next Steps

1. Clone the repo and set up API credentials
2. Build a strategy plugin for arbitrage detection
3. Test with dry-run mode
4. Deploy with proper key management
