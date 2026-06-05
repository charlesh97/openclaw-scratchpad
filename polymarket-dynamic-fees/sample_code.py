#!/usr/bin/env python3
"""
Fee-aware arb calculator for Polymarket 15-minute markets.
Accounts for dynamic taker fees to determine true edge.
"""

def get_dynamic_fee(odds: float) -> float:
    """
    Dynamic taker fee based on how close odds are to 50%.
    Highest at 50/50 (~3.15% of 50¢ = ~1.575¢ per contract).
    Lower as odds diverge from 50/50.
    """
    # Approximate fee curve based on Polymarket's design
    distance_from_50 = abs(odds - 0.50)
    if distance_from_50 < 0.05:  # 45-55¢ range
        return 0.0315 * odds  # ~3.15%
    elif distance_from_50 < 0.15:  # 35-45 or 55-65¢
        return 0.0210 * odds
    elif distance_from_50 < 0.25:  # 25-35 or 65-75¢
        return 0.0105 * odds
    else:
        return 0.0025 * odds  # minimal

def calculate_net_arb_edge(yes_price: float, no_price: float, is_15m: bool = True) -> dict:
    """Calculate true arb edge after fees."""
    total_cost = yes_price + no_price
    
    if is_15m:
        # Taker fee applies to the purchase
        fee_yes = get_dynamic_fee(yes_price)
        fee_no = get_dynamic_fee(no_price)
        total_fees = fee_yes + fee_no
    else:
        total_fees = 0.0
    
    net_cost = total_cost + total_fees
    gross_edge = 1.0 - total_cost
    net_edge = 1.0 - net_cost
    
    return {
        "gross_edge": gross_edge,
        "net_edge": net_edge,
        "fees": total_fees,
        "profitable": net_edge > 0.0
    }

# Example: 48¢ YES + 49¢ NO on 15-min market
result = calculate_net_arb_edge(0.48, 0.49, is_15m=True)
print(f"Gross edge: {result['gross_edge']:.2%}")
print(f"Fees: {result['fees']:.4f}¢")
print(f"Net edge: {result['net_edge']:.2%}")
print(f"Profitable: {result['profitable']}")
