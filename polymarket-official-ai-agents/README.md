# Polymarket Official AI Agents Framework

**Source:** [github.com/Polymarket/agents](https://github.com/Polymarket/agents)

## What It Does

Polymarket's official developer framework and utility library for building **AI agents that trade autonomously** on Polymarket. Provides the official API infrastructure for agent-based trading strategies.

## Key Features

- **MCP Server integration** — Model Context Protocol server enabling AI agents (Claude, etc.) to interact with Polymarket
- **45+ tools** for market data, order management, portfolio tracking, and strategy execution
- **Real-time monitoring** with enterprise-grade safety features
- **Official SDK** — maintained by Polymarket team, ensuring API compatibility
- **Autonomous trading** — agents can discover markets, evaluate probabilities, and execute trades without human intervention

## Why It Matters

This is the **official first-party framework** from Polymarket. It guarantees:
- Longest API compatibility window
- Access to new Polymarket features first
- Best documentation and community support
- Direct integration with Polymarket's order book, CTF, and data APIs

For bot development, this is the infrastructure layer — not a strategy itself, but the foundation any strategy gets built on.

## Risks

- Framework is in active development (may have breaking changes)
- Relies on Polymarket's API infrastructure (rate limits, downtime)
- No built-in trading strategies — you provide your own
- AI agent integration adds latency vs. direct API calls

## Implementability: 4/5

Production-ready framework. The MCP integration is particularly valuable for AI-driven strategy research and rapid prototyping.

## Next Steps

1. Set up MCP server with Claude for interactive strategy development
2. Build a prototype agent that executes one of our existing strategies
3. Test latency vs. direct CLOB API calls
4. Evaluate the 45 tools for arbitrage-specific use cases
