"""
ImMike/polymarket-arbitrage — Sample Cross-Platform Arbitrage Logic
Based on: https://github.com/ImMike/polymarket-arbitrage

Simplified pseudocode showing how cross-platform arbitrage is detected.
"""
import requests

POLYMARKET_API = "https://gamma-api.polymarket.com"
KALSHI_API = "https://api.kalshi.com"

def fetch_polymarket_markets():
    """Fetch all active Polymarket markets."""
    r = requests.get(f"{POLYMARKET_API}/markets", params={"closed": "false"})
    return r.json()

def fetch_kalshi_markets():
    """Fetch all active Kalshi markets."""
    r = requests.get(f"{KALSHI_API}/events", params={"status": "open"})
    return r.json()

def normalize_price(price_cents):
    """Convert Kalshi cents to probability float."""
    return price_cents / 100.0

def check_cross_platform_arb(poly_yes_bid, kalshi_yes_bid, min_edge=0.01):
    """
    Detect arb where buying YES on Polymarket and selling YES on Kalshi
    (or vice versa) yields profit after fees.

    poly_yes_bid: float 0-1 (Polymarket best bid for YES)
    kalshi_yes_bid: float 0-1 (Kalshi best bid for YES)
    """
    edge = abs(poly_yes_bid - kalshi_yes_bid)
    if edge < min_edge:
        return None  # No enough edge after fees

    if poly_yes_bid < kalshi_yes_bid:
        # Strategy: Buy on Polymarket (cheap), Sell on Kalshi (expensive)
        return {
            "strategy": "BUY_POLY_YES_SELL_KALSHI_YES",
            "buy_platform": "polymarket",
            "sell_platform": "kalshi",
            "edge": edge,
        }
    else:
        # Strategy: Buy on Kalshi (cheap), Sell on Polymarket (expensive)
        return {
            "strategy": "BUY_KALSHI_YES_SELL_POLY_YES",
            "buy_platform": "kalshi",
            "sell_platform": "polymarket",
            "edge": edge,
        }

def check_bundle_arb(yes_ask, no_ask, min_edge=0.01):
    """
    Detect bundle arbitrage: YES + NO costs less than $1.00.
    When ask_yes + ask_no < $1.00, buy both → guaranteed $1 payout.

    yes_ask: float (cost to buy YES)
    no_ask: float (cost to buy NO)
    """
    total_cost = yes_ask + no_ask
    edge = 1.0 - total_cost
    if edge >= min_edge:
        return {
            "strategy": "BUNDLE_BUY_BOTH",
            "cost": total_cost,
            "edge": edge,
            "profit_per_dollar": edge / total_cost,
        }
    return None

if __name__ == "__main__":
    # Example: Trump win prediction
    poly_yes = 0.52  # Polymarket YES bid
    kalshi_yes = 0.58  # Kalshi YES bid

    result = check_cross_platform_arb(poly_yes, kalshi_yes)
    print(f"Cross-platform arb: {result}")
    # → {'strategy': 'BUY_POLY_YES_SELL_KALSHI_YES', 'edge': 0.06}
