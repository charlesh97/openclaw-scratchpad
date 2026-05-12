"""
PIRAP Framework — Conceptual components (from arXiv:2605.10400).

This is a high-level pseudocode representation of the six PIRAP components.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


# === Component 1: Index Estimator ===

@dataclass
class IndexEstimate:
    mid_price: float
    depth_weighted_mid: float
    time_decayed_vwap: float
    composite_index: float


def compute_index(order_book: dict, trades: List[dict], 
                  decay_half_life: float = 60.0) -> IndexEstimate:
    """Combine three estimators into a composite index."""
    bids = order_book['bids']  # [[price, size], ...]
    asks = order_book['asks']
    
    # 1. Mid-price
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    
    # 2. Depth-weighted mid (signed volume to 2% depth)
    depth_bids = sum(s for p, s in bids if p >= best_bid * 0.98)
    depth_asks = sum(s for p, s in asks if p <= best_ask * 1.02)
    dwm = (best_bid * depth_asks + best_ask * depth_bids) / (depth_bids + depth_asks)
    
    # 3. Time-decayed VWAP
    decay = np.exp(-np.arange(len(trades)) / decay_half_life)
    w = decay / decay.sum()
    td_vwap = sum(t['price'] * w[i] for i, t in enumerate(trades))
    
    # Composite: equal weight for simplicity
    composite = (mid + dwm + td_vwap) / 3.0
    
    return IndexEstimate(mid, dwm, td_vwap, composite)


# === Component 2: Jump-Aware Tiered Margin ===

def compute_margin(collateral: float, probability: float,
                   time_to_resolution: float, volatility: float) -> float:
    """
    Margin requirement increases as resolution approaches,
    accounting for terminal-collapse jumps.
    """
    base_margin = collateral * 0.1  # 10% base
    
    # Jump component: max 50% swing near resolution
    jump_magnitude = 0.5 * np.exp(-time_to_resolution / 3600)  # decays over hours
    
    # Volatility component
    vol_margin = volatility * collateral * 0.5
    
    total_margin = base_margin + jump_magnitude * collateral + vol_margin
    return min(total_margin, collateral * 0.5)  # capped at 50%


# === Component 3: Leverage Compression ===

def leverage_compression_schedule(time_to_resolution: float,
                                   initial_leverage: float = 10.0) -> float:
    """
    Leverage compresses from initial_max → 1x linearly as resolution approaches.
    """
    if time_to_resolution <= 0:
        return 1.0
    
    max_time = 7 * 24 * 3600  # 7 days in seconds
    progress = 1.0 - min(time_to_resolution / max_time, 1.0)
    return max(1.0, initial_leverage * (1.0 - progress))


# === Component 4: Resolution-Aware Funding ===

def compute_funding_rate(index_price: float, mark_price: float,
                         probability: float, time_to_resolution: float) -> float:
    """
    Funding rate with boundary-aware correction.
    Near boundaries (p ≈ 0 or p ≈ 1), funding adjusts to prevent manipulation.
    """
    basis = mark_price - index_price
    
    # Standard funding
    base_funding = basis / index_price if index_price > 0 else 0.0
    
    # Boundary correction — dampens funding near 0 or 1
    boundary_factor = 4.0 * probability * (1.0 - probability)  # 0 at edges, 1 at 0.5
    corrected_funding = base_funding * boundary_factor
    
    return corrected_funding


# === Component 5: Halt Protocol ===

def should_halt(price_change_pct: float, time_to_resolution: float,
                consecutive_violations: int) -> bool:
    """Multi-stage halt: triggers at increasing thresholds when close to resolution."""
    threshold = 0.10 * (1.0 + consecutive_violations)  # 10%, 20%, 30%...
    if time_to_resolution < 3600:  # last hour
        threshold *= 0.5  # more sensitive
    
    return abs(price_change_pct) > threshold


# === Component 6: Eligibility ===

def is_eligible_market(volume_24h: float, spread_bps: float,
                       depth_usdc: float, age_days: int) -> bool:
    """Minimum eligibility gates for a market to be an underlying."""
    return (
        volume_24h > 10000         # $10k+ daily volume
        and spread_bps < 100       # < 1% spread
        and depth_usdc > 5000      # $5k+ order book depth
        and age_days > 1           # at least 1 day old
    )
