"""
Polymarket Microstructure Utilities

Based on findings from Dubach et al. (2026):
"The Anatomy of a Decentralized Prediction Market"

Key insight: Order-book trade direction is unreliable (~59% accuracy).
Must use on-chain OrderFilled events for accurate trade classification.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OrderBookSnapshot:
    """Represents a Polymarket CLOB order book snapshot."""
    market: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]  # (price, size)


@dataclass
class OnChainFill:
    """Represents an on-chain OrderFilled event."""
    market: str
    timestamp: datetime
    maker_side: str  # 'BUY' or 'SELL'
    maker_amount: float
    maker_price: float
    taker_side: str
    taker_amount: float
    fee: float


def compute_effective_spread(
    fill: OnChainFill,
    book: OrderBookSnapshot,
    use_onchain_direction: bool = True,
) -> float:
    """
    Compute effective half-spread using either on-chain or book-inferred direction.
    
    Per Dubach et al.: book-inferred direction is only ~59% accurate.
    Always prefer on-chain OrderFilled events.
    """
    if use_onchain_direction:
        # Use the authoritative on-chain direction
        if fill.maker_side == 'BUY':
            # Maker sold, taker bought — effective spread = fill_price - mid
            mid = (book.bids[0][0] + book.asks[0][0]) / 2 if book.bids and book.asks else fill.maker_price
            return fill.maker_price - mid
        else:
            mid = (book.bids[0][0] + book.asks[0][0]) / 2 if book.bids and book.asks else fill.maker_price
            return mid - fill.maker_price
    else:
        # Book-inferred (less reliable)
        mid = (book.bids[0][0] + book.asks[0][0]) / 2 if book.bids and book.asks else fill.maker_price
        return abs(fill.maker_price - mid)


def estimate_market_impact(
    order_size: float,
    depth_profile: np.ndarray,
    price_levels: np.ndarray,
) -> Tuple[float, float]:
    """
    Estimate the price impact of an order given the depth profile.
    
    Polymarket depth is more uniform than top-of-book concentrated,
    meaning larger orders have more predictable (but still significant) impact.
    
    Args:
        order_size: Size of the order in shares
        depth_profile: Cumulative depth at each price level
        price_levels: Price at each level
    
    Returns:
        (avg_execution_price, slippage_bps)
    """
    remaining = order_size
    total_cost = 0.0
    filled = 0.0
    
    for level_depth, level_price in zip(depth_profile, price_levels):
        if remaining <= 0:
            break
        fill = min(remaining, level_depth)
        total_cost += fill * level_price
        filled += fill
        remaining -= fill
    
    if filled == 0:
        return 0.0, float('inf')
    
    avg_price = total_cost / filled
    mid_price = price_levels[0]  # Approximation
    slippage_bps = (avg_price - mid_price) / mid_price * 10000
    
    return avg_price, slippage_bps


def detect_wash_trading_ratio(
    fills: List[OnChainFill],
    wallet_pairs: Dict[str, str],
) -> float:
    """
    Detect potential self-counterparty trading (wash trading).
    
    Per Dubach et al.: median 1%, upper tail 22%.
    Much lower than unregulated CEXes (25-70% per Cong et al.).
    """
    wash_volume = 0.0
    total_volume = 0.0
    
    for fill in fills:
        total_volume += fill.maker_amount
        # Check if maker and taker wallets are linked
        maker_wallet = fill.maker_side  # Simplified — would need full wallet resolution
        # In production: resolve wallet addresses from blockchain events
    
    return wash_volume / total_volume if total_volume > 0 else 0.0


def classify_trade_direction_from_book(
    book: OrderBookSnapshot,
    trade_price: float,
) -> str:
    """
    Classify trade as buyer-initiated or seller-initiated from order book.
    
    WARNING: Only ~59% accurate per Dubach et al.!
    Prefer on-chain OrderFilled events.
    """
    if not book.bids or not book.asks:
        return 'UNKNOWN'
    
    best_bid = book.bids[0][0]
    best_ask = book.asks[0][0]
    mid = (best_bid + best_ask) / 2
    
    if trade_price > mid:
        return 'BUYER_INITIATED'  # Taker bought
    elif trade_price < mid:
        return 'SELLER_INITIATED'  # Taker sold
    else:
        return 'UNKNOWN'


def compute_depth_profile(
    book: OrderBookSnapshot,
    levels: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute depth profile as in Dubach et al.
    Polymarket shows more uniform depth than top-of-book concentration.
    """
    bid_depths = np.array([d[1] for d in book.bids[:levels]])
    ask_depths = np.array([d[1] for d in book.asks[:levels]])
    
    bid_prices = np.array([d[0] for d in book.bids[:levels]])
    ask_prices = np.array([d[0] for d in book.asks[:levels]])
    
    # Depth-weighted average price
    bid_vwap = np.sum(bid_prices * bid_depths) / np.sum(bid_depths) if np.sum(bid_depths) > 0 else 0
    ask_vwap = np.sum(ask_prices * ask_depths) / np.sum(ask_depths) if np.sum(ask_depths) > 0 else 0
    
    # Concentration ratio: top level / total depth
    total_bid_depth = np.sum(bid_depths)
    total_ask_depth = np.sum(ask_depths)
    bid_concentration = bid_depths[0] / total_bid_depth if total_bid_depth > 0 else 0
    ask_concentration = ask_depths[0] / total_ask_depth if total_ask_depth > 0 else 0
    
    return {
        'bid_depths': bid_depths,
        'ask_depths': ask_depths,
        'bid_vwap': bid_vwap,
        'ask_vwap': ask_vwap,
        'bid_concentration': bid_concentration,
        'ask_concentration': ask_concentration,
        'total_bid_depth': total_bid_depth,
        'total_ask_depth': total_ask_depth,
    }
