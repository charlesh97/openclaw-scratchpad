"""
Python reference: Automated Market Making (Poly-Maker style)
Based on terrytrl100/polymarket-automated-mm / @defiance_cr's Poly-Maker

Core logic: two-sided quoting with reward-optimized pricing.
"""

import asyncio
import time
import json
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketState:
    condition_id: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    mid_price: float
    volume_24h: float
    daily_reward_est: float
    volatility: float


@dataclass
class Quote:
    outcome: str  # "YES" or "NO"
    side: str     # "buy" or "sell"
    price: float
    size: int
    reason: str


class RewardOptimizedPricer:
    """
    Calculate optimal quote prices based on Polymarket's maker reward formula.

    Polymarket pays makers a reward proportional to:
    - Order book depth provided
    - Time at the top of the book
    - Spread tightness
    """

    def __init__(self, target_spread_bps: float = 20.0, min_profit_bps: float = 5.0):
        self.target_spread_bps = target_spread_bps
        self.min_profit_bps = min_profit_bps

    def calculate_quotes(self, market: MarketState) -> tuple[Quote, Quote]:
        """Generate optimal two-sided quotes for a market."""

        mid = market.mid_price
        half_spread = mid * self.target_spread_bps / 10000 / 2

        # Buy below fair value, sell above
        buy_price = max(0.001, mid - half_spread)
        sell_price = min(0.999, mid + half_spread)

        # Dynamic sizing based on volume and volatility
        base_size = max(10, int(market.volume_24h / 1000))
        vol_adj = max(0.5, 1.0 - market.volatility * 2)  # reduce size when volatile
        size = int(base_size * vol_adj)

        buy_quote = Quote(
            outcome="YES",
            side="buy",
            price=round(buy_price, 4),
            size=size,
            reason=f"mid={mid:.4f}, spread={self.target_spread_bps}bps"
        )

        sell_quote = Quote(
            outcome="YES",
            side="sell",
            price=round(sell_price, 4),
            size=size,
            reason=f"mid={mid:.4f}, spread={self.target_spread_bps}bps"
        )

        return buy_quote, sell_quote


class MarketSelector:
    """Data-driven market selection based on profitability."""

    def __init__(self, min_daily_reward: float = 50.0, max_markets: int = 10):
        self.min_daily_reward = min_daily_reward
        self.max_markets = max_markets

    def select_markets(self, all_markets: list[MarketState]) -> list[MarketState]:
        """Rank and select the most profitable markets."""

        # Filter markets above minimum reward
        eligible = [m for m in all_markets if m.daily_reward_est >= self.min_daily_reward]

        # Sort by reward/volatility ratio (higher = better)
        scored = sorted(
            eligible,
            key=lambda m: m.daily_reward_est / max(m.volatility, 0.01),
            reverse=True
        )

        return scored[:self.max_markets]


class OrderManager:
    """Manage active orders and positions."""

    def __init__(self):
        self.active_orders: dict[str, Quote] = {}
        self.positions: dict[str, int] = {}
        self.gas_cost: float = 0.0001  # ~$0.0001 per tx on Polygon

    def should_cancel(self, old: Quote, new: Quote) -> bool:
        """Decide if price moved enough to justify cancellation and replacement."""
        price_diff = abs(old.price - new.price) / old.price
        return price_diff > 0.005  # cancel if > 50bps moved

    def can_open_position(self, market_id: str, max_positions: int = 20) -> bool:
        """Check if we can take on another position."""
        return len(self.active_orders) < max_positions


class PolyMakerBot:
    """Main market making bot loop."""

    def __init__(self, bankroll_usdc: float = 10000.0):
        self.bankroll = bankroll_usdc
        self.pricer = RewardOptimizedPricer()
        self.selector = MarketSelector(min_daily_reward=50.0, max_markets=5)
        self.orders = OrderManager()

    async def run_cycle(self):
        """One cycle: fetch markets → select → quote → place orders."""
        # 1. Fetch all markets (would call Polymarket API)
        all_markets = await self._fetch_markets()

        # 2. Select best markets
        selected = self.selector.select_markets(all_markets)

        # 3. Generate quotes for each selected market
        for market in selected:
            buy_q, sell_q = self.pricer.calculate_quotes(market)

            key = f"{market.condition_id}_YES"

            # Check if existing quote needs updating
            if key in self.orders.active_orders:
                existing = self.orders.active_orders[key]
                if self.orders.should_cancel(existing, buy_q):
                    await self._cancel_order(existing)
                else:
                    continue  # skip — existing quote is still good

            # Place new order
            await self._place_order(buy_q)
            await self._place_order(sell_q)
            self.orders.active_orders[key] = buy_q

        # 4. Log status
        print(f"Cycle complete: {len(selected)} markets, "
              f"{len(self.orders.active_orders)} active orders, "
              f"bankroll: ${self.bankroll:.2f}")

    async def _fetch_markets(self) -> list[MarketState]:
        """Placeholder — in production calls Polymarket Gamma API."""
        return [
            MarketState(
                condition_id=f"0x{i:x}",
                yes_bid=0.45, yes_ask=0.55,
                no_bid=0.42, no_ask=0.52,
                mid_price=0.50,
                volume_24h=50000,
                daily_reward_est=75.0 + i * 10,
                volatility=0.15 + i * 0.01,
            )
            for i in range(20)
        ]

    async def _place_order(self, quote: Quote):
        """Placeholder — in production calls Polymarket CLOB API."""
        print(f"  PLACE: {quote.side} {quote.outcome} @ {quote.price} x {quote.size} "
              f"({quote.reason})")

    async def _cancel_order(self, quote: Quote):
        """Placeholder — in production calls Polymarket CLOB API."""
        print(f"  CANCEL: {quote.side} {quote.outcome} @ {quote.price}")


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def main():
        bot = PolyMakerBot(bankroll_usdc=10000)
        # Run 3 cycles with 30-second intervals
        for i in range(3):
            print(f"\n=== Cycle {i+1} ===")
            await bot.run_cycle()
            await asyncio.sleep(30)

    asyncio.run(main())
