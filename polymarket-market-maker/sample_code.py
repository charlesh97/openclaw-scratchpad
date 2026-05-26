"""
Reference implementation: Market making patterns from the Paradigm #2 strategy
(octavi42/prediction-market-maker)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Quote:
    side: str  # "bid" or "ask"
    price: float
    size: int


class RegimeAwareMarketMaker:
    """
    Market maker that detects monopoly regime (being the only quote on a side)
    and adjusts quoting behavior accordingly.
    """

    def __init__(self, fair_price: float, volatility: float):
        self.fair_price = fair_price
        self.volatility = volatility
        self.base_spread = 0.02  # 2 cents
        self.monopoly_spread_mult = 2.5  # wider spread when alone on side
        self.inventory_skew = 0.0

    def compute_quotes(
        self, book_state: dict, inventory: float
    ) -> tuple[Quote, Quote]:
        """
        Compute optimal bid/ask quotes based on book state and inventory.
        """
        my_bids = book_state.get("my_bids", [])
        my_asks = book_state.get("my_asks", [])
        other_bids = book_state.get("other_bids", [])
        other_asks = book_state.get("other_asks", [])

        am_i_alone_on_bid = len(other_bids) == 0 and len(my_bids) == 0
        am_i_alone_on_ask = len(other_asks) == 0 and len(my_asks) == 0

        # Volatility-adjusted spread — tighter when volatile
        vol_adjustment = max(0.5, 1.0 - self.volatility * 10)
        spread = self.base_spread * vol_adjustment

        # Monopoly regime — widen spread
        bid_spread = spread * (
            self.monopoly_spread_mult if am_i_alone_on_bid else 1.0
        )
        ask_spread = spread * (
            self.monopoly_spread_mult if am_i_alone_on_ask else 1.0
        )

        # Inventory skew — move quotes to reduce directional risk
        inventory_adjust = inventory * self.inventory_skew

        bid_price = self.fair_price - bid_spread / 2 + inventory_adjust
        ask_price = self.fair_price + ask_spread / 2 + inventory_adjust

        bid_price = np.clip(bid_price, 0.01, 0.99)
        ask_price = np.clip(ask_price, 0.01, 0.99)

        return (
            Quote(side="bid", price=round(bid_price, 2), size=10),
            Quote(side="ask", price=round(ask_price, 2), size=10),
        )
