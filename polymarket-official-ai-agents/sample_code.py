#!/usr/bin/env python3
"""
Sample: Using Polymarket/agents framework for an arbitrage detection agent.
"""
import os
from connectors.polymarket import Polymarket
from connectors.gamma import GammaMarketClient

# Initialize (requires POLYGON_WALLET_PRIVATE_KEY + OPENAI_API_KEY in .env)
pm = Polymarket()
gamma = GammaMarketClient()

# Fetch all tradable markets
markets = gamma.get_current_markets(limit=50)

# Scan for bundle arb opportunities (YES + NO < $1)
for m in markets:
    yes_bid = m.get("yes_bid", 0)
    no_bid = m.get("no_bid", 0)
    total = yes_bid + no_bid
    
    if total < 0.98:
        edge = 1.0 - total
        print(f"ARB: {m['question']} | YES={yes_bid:.2f} NO={no_bid:.2f} | "
              f"Total={total:.2f} | Edge={edge:.2%}")

# Execute trade via CLI
# python scripts/python/cli.py trade --market <token_id> --side BUY --size 10
