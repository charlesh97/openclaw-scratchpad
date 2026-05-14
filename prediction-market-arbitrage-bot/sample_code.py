"""
Reference: https://github.com/realfishsam/prediction-market-arbitrage-bot

Simplified cross-platform arbitrage scanner using pmxt.dev-style abstraction.
"""

from dataclasses import dataclass
from typing import Optional, List
import time


@dataclass
class MarketQuote:
    platform: str       # "polymarket" or "kalshi"
    market_id: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume_24h: float


@dataclass
class ArbSignal:
    market_title: str
    buy_platform: str
    sell_platform: str
    buy_price: float
    sell_price: float
    gross_spread: float
    net_spread_after_fees: float
    max_size: int


class CrossPlatformArbScanner:
    """
    Scans both Polymarket and Kalshi for identical markets,
    then identifies profitable cross-platform arbitrage opportunities.
    """

    def __init__(self, polymarket_fee: float = 0.01, kalshi_fee: float = 0.015):
        self.poly_fee = polymarket_fee
        self.kalshi_fee = kalshi_fee
        self.price_cache: dict = {}  # market_title -> {platform: quote}

    def find_matching_markets(
        self, poly_quotes: List[MarketQuote], kalshi_quotes: List[MarketQuote]
    ) -> List[ArbSignal]:
        """
        Match markets by title across platforms and compute arb spreads.
        """
        # Build a lookup from normalized title
        kalshi_map = {}
        for q in kalshi_quotes:
            key = self._normalize_title(q.market_id)
            kalshi_map[key] = q

        signals = []
        for pq in poly_quotes:
            key = self._normalize_title(pq.market_id)
            kq = kalshi_map.get(key)
            if not kq:
                continue

            # Detect price discrepancies
            # Scenario: Buy YES on cheaper platform, sell (or buy NO) on expensive
            if pq.yes_ask < kq.yes_bid:
                spread = (kq.yes_bid - pq.yes_ask) / pq.yes_ask
                fees = self.poly_fee + self.kalshi_fee
                net_spread = spread - fees
                if net_spread > 0:
                    signals.append(ArbSignal(
                        market_title=key,
                        buy_platform="polymarket",
                        sell_platform="kalshi",
                        buy_price=pq.yes_ask,
                        sell_price=kq.yes_bid,
                        gross_spread=spread,
                        net_spread_after_fees=net_spread,
                        max_size=min(pq.volume_24h, kq.volume_24h) // 100,
                    ))

            # Reverse: Kalshi cheaper
            if kq.yes_ask < pq.yes_bid:
                spread = (pq.yes_bid - kq.yes_ask) / kq.yes_ask
                fees = self.poly_fee + self.kalshi_fee
                net_spread = spread - fees
                if net_spread > 0:
                    signals.append(ArbSignal(
                        market_title=key,
                        buy_platform="kalshi",
                        sell_platform="polymarket",
                        buy_price=kq.yes_ask,
                        sell_price=pq.yes_bid,
                        gross_spread=spread,
                        net_spread_after_fees=net_spread,
                        max_size=min(pq.volume_24h, kq.volume_24h) // 100,
                    ))

        return sorted(signals, key=lambda s: s.net_spread_after_fees, reverse=True)

    def _normalize_title(self, title: str) -> str:
        """Normalize market title for cross-platform matching."""
        return title.lower().strip().replace(" ", "").replace("-", "").replace("_", "")


# Example: Risk-free return calculation
def compute_risk_free_return(signal: ArbSignal, capital: float) -> dict:
    """
    Compute expected return and max position for a cross-platform arb.
    """
    min_lot = 1  # Kalshi = 1 contract = $1; Polymarket = 1 share
    max_buy = int(capital / signal.buy_price)

    # Fills are not guaranteed; apply a fill penalty factor
    expected_fill_rate = 0.85

    gross_return = max_buy * (signal.sell_price - signal.buy_price)
    net_return = gross_return * expected_fill_rate

    return {
        "capital_deployed": max_buy * signal.buy_price,
        "max_contracts": max_buy,
        "gross_return": gross_return,
        "expected_return": net_return,
        "fill_rate_assumption": expected_fill_rate,
        "roi": net_return / (max_buy * signal.buy_price) * 100,
    }
