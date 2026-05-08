"""
CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot — Bitcoin Hourly Arbitrage Logic
Based on: https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot

For the BTC 1-Hour Price market: if Poly Down + Kalshi Yes costs < $1.00,
buy both → guaranteed ~$1 payout regardless of BTC direction.
"""
import requests

def calculate_btc_arb(poly_down_bid, kalshi_yes_bid):
    """
    poly_down_bid: Polymarket best bid for "Bitcoin DOWN" (0-1)
    kalshi_yes_bid: Kalshi best bid for "Bitcoin UP" (0-1)
    """
    total_cost = poly_down_bid + kalshi_yes_bid
    if total_cost < 1.00:
        edge = 1.00 - total_cost
        roi = edge / total_cost * 100
        return {
            "legs": "BUY Poly DOWN + BUY Kalshi YES",
            "total_cost": f"${total_cost:.2f}",
            "guaranteed_payout": "$1.00",
            "edge": f"${edge:.2f}",
            "roi_percent": f"{roi:.1f}%",
            "executable": True,
        }
    return {"executable": False, "reason": "total_cost >= $1.00"}

# Example
print(calculate_btc_arb(0.47, 0.48))
# → total_cost = 0.95, edge = $0.05, roi = 5.3%
