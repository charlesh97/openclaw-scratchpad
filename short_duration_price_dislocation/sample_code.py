#!/usr/bin/env python3
"""
Short-Duration Price Dislocation Arbitrage — Reference Implementation
=====================================================================
Exploits moments when YES + NO combined price dips below $1.00 in short-duration
Polymarket Up/Down binary contracts (5–15 minute expiry).

Strategy: When (price_YES + price_NO) < 1.00 − fees, buy both sides.
         At settlement, one side pays $1.00 — guaranteed profit.

Key requirements:
  - WebSocket streaming for sub-second order book updates
  - Fee-aware edge calculation (Polymarket taker fees: 0.75–1.80%)
  - Small position sizing to avoid slippage (max ~$2K per round-trip)
  - Maker-order preference; taker only when edge is large

This is a REFERENCE implementation for educational purposes.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------

@dataclass
class MarketState:
    """Snapshot of a short-duration binary market's order book state."""
    market_id: str
    yes_bid: float    # best YES bid (what you can sell at)
    yes_ask: float   # best YES ask  (what you can buy at)
    no_bid: float    # best NO bid
    no_ask: float    # best NO ask
    yes_depth: float # approximate depth on YES side ($)
    no_depth: float  # approximate depth on NO side ($)
    timestamp: float


@dataclass
class ArbOpportunity:
    """A within-platform dislocation opportunity."""
    market_id: str
    edge_cents: float          # gross edge in cents (1.00 - combined_cost)
    fee_adjusted_edge: float  # edge after platform fees
    combined_cost: float      # yes_ask + no_ask
    max_leg_size: float       # smallest depth-limited leg
    leg_currency: str = "USDC"
    timestamp: float = 0.0


# ------------------------------------------------------------------------------
# Fee Calculator
# ------------------------------------------------------------------------------

def polymarket_taker_fee_rate(category: str = "crypto") -> float:
    """
    Polymarket taker fee tiers (as of 2026).
    Fees reduce effective edge — only trade when fee-adjusted edge > 0.
    """
    # Fee is a fraction (0.01 = 1%)
    tiers = {
        "crypto":    0.0175,  # ~1.75% for crypto markets (midpoint of 0.75–1.80%)
        "political": 0.0125,  # ~1.25%
        "sports":    0.0180,  # ~1.80%
        "default":   0.0150,  # conservative default
    }
    return tiers.get(category, tiers["default"])


def calc_edge(state: MarketState, fee_rate: float = 0.0175) -> Optional[ArbOpportunity]:
    """
    Calculate whether a within-platform dislocation exists.

    Using ASK prices (what you pay to enter):
      combined_cost = yes_ask + no_ask
      gross_edge    = 1.00 - combined_cost

    Fee-adjusted edge accounts for the fact that closing both legs
    also incurs fees on each platform when selling YES/NO back.
    For a simple buy-both-hold-to-settlement arb, each leg pays
    a taker fee on entry. Exit at settlement is $1 received, no fee.

    More conservative: apply fee on combined_cost.
      fee_adjusted_edge = 1.00 - combined_cost * (1 + fee_rate)
    """
    combined_cost = state.yes_ask + state.no_ask

    # Gross edge: combined_cost below $1 means arb exists
    gross_edge = 1.00 - combined_cost

    # Conservative fee adjustment: apply fee to each leg at entry
    fee_adjustment = combined_cost * fee_rate
    fee_adjusted_edge = gross_edge - fee_adjustment

    # Max leg size limited by order book depth on each side
    max_leg_size = min(state.yes_depth, state.no_depth)

    if fee_adjusted_edge > 0 and max_leg_size > 0:
        return ArbOpportunity(
            market_id=state.market_id,
            edge_cents=gross_edge * 100,
            fee_adjusted_edge=fee_adjusted_edge * 100,
            combined_cost=combined_cost,
            max_leg_size=max_leg_size,
            timestamp=state.timestamp,
        )
    return None


# ------------------------------------------------------------------------------
# Signal Engine
# ------------------------------------------------------------------------------

class ShortDurationArbEngine:
    """
    Scanning engine for short-duration Polymarket price dislocations.

    Usage:
        engine = ShortDurationArbEngine()
        engine.add_callback(handle_opportunity)
        asyncio.get_event_loop().run_until_complete(engine.run())
    """

    def __init__(
        self,
        min_edge_cents: float = 0.50,    # minimum fee-adj edge (cents) to trigger
        max_position_usd: float = 2000,  # max position per leg
        fee_rate: float = 0.0175,
        scan_interval_ms: int = 50,        # scan every 50ms for millisecond windows
    ):
        self.min_edge_cents = min_edge_cents
        self.max_position_usd = max_position_usd
        self.fee_rate = fee_rate
        self.scan_interval_ms = scan_interval_ms
        self.callbacks: list = []

    def add_callback(self, fn):
        """Register a callback(oPPORTUNITY) to run when arb is detected."""
        self.callbacks.append(fn)

    async def scan_market(self, state: MarketState) -> Optional[ArbOpportunity]:
        """Evaluate a single market state for dislocation."""
        opp = calc_edge(state, self.fee_rate)
        if opp and opp.fee_adjusted_edge >= self.min_edge_cents:
            # Cap position at max_position_usd and book depth
            capped_size = min(self.max_position_usd, opp.max_leg_size)
            if capped_size < 10:
                return None  # too small to be worth it
            return opp
        return None

    async def run(self, market_ids: list[str], data_source):
        """
        Main scanning loop. `data_source` must implement:
            data_source.stream_order_book(market_id) -> AsyncIterator[MarketState]
        """
        print(f"[ShortDurationArb] Started — scanning {len(market_ids)} markets every {self.scan_interval_ms}ms")
        print(f"[ShortDurationArb] Min edge: {self.min_edge_cents}c | Max position: ${self.max_position_usd}")

        tasks = [
            self._scan_loop(market_id, data_source)
            for market_id in market_ids
        ]
        await asyncio.gather(*tasks)

    async def _scan_loop(self, market_id: str, data_source):
        """Scan a single market continuously."""
        async for state in data_source.stream_order_book(market_id):
            opp = await self.scan_market(state)
            if opp:
                for cb in self.callbacks:
                    asyncio.create_task(self._safe_callback(cb, opp))

    async def _safe_callback(self, cb, opp):
        try:
            if asyncio.iscoroutinefunction(cb):
                await cb(opp)
            else:
                cb(opp)
        except Exception as e:
            print(f"[ShortDurationArb] Callback error: {e}")


# ------------------------------------------------------------------------------
# Mock Data Source (for backtesting / development)
# ------------------------------------------------------------------------------

class MockPolymarketWS:
    """
    Mock WebSocket data source. Replace with real Gamma API / Polymarket CLOB client.
    In production, use the Polymarket API or a WebSocket feed with order-book data.

    Usage:
        source = MockPolymarketWS(market_ids=["btc-5min-up", "eth-5min-up"])
        async for state in source.stream_order_book("btc-5min-up"):
            ...
    """

    def __init__(self, market_ids: list[str], seed: int = 42):
        import random
        self.market_ids = market_ids
        self.rng = random.Random(seed)
        self.base_prices = {mid: 0.50 for mid in market_ids}
        self.order_books = {mid: self._gen_ob() for mid in market_ids}

    def _gen_ob(self):
        """Generate a random order book around fair value."""
        fair = 0.50
        spread = 0.005
        return {
            "yes_bid":  fair - spread,
            "yes_ask":  fair + spread,
            "no_bid":   1 - (fair + spread),
            "no_ask":   1 - (fair - spread),
            "yes_depth": self.rng.uniform(2000, 15000),
            "no_depth":  self.rng.uniform(2000, 15000),
        }

    async def stream_order_book(self, market_id: str):
        """Yield simulated order book snapshots at ~20Hz."""
        while True:
            ob = self.order_books.get(market_id)
            if not ob:
                raise StopAsyncIteration

            # Occasionally inject a dislocation event (1% chance per tick)
            if self.rng.random() < 0.01:
                dislocation_depth = self.rng.uniform(0.90, 0.99)
                combined = dislocation_depth
                ob = {
                    "yes_bid":  combined * 0.49,
                    "yes_ask":  combined * 0.51,
                    "no_bid":   (1 - combined) * 0.49,
                    "no_ask":   (1 - combined) * 0.51,
                    "yes_depth": self.rng.uniform(1000, 5000),
                    "no_depth":  self.rng.uniform(1000, 5000),
                }

            yield MarketState(
                market_id=market_id,
                yes_bid=ob["yes_bid"],
                yes_ask=ob["yes_ask"],
                no_bid=ob["no_bid"],
                no_ask=ob["no_ask"],
                yes_depth=ob["yes_depth"],
                no_depth=ob["no_depth"],
                timestamp=time.time(),
            )
            await asyncio.sleep(0.05)  # 20Hz


# ------------------------------------------------------------------------------
# Order Execution (skeleton — requires real API integration)
# ------------------------------------------------------------------------------

class ExecutionRouter:
    """
    Skeleton execution router for Polymarket CLOB.
    Replace with real API calls to Polymarket Gamma API.

    Key decisions:
      - Maker preferred: place limit orders to earn rebate
      - Taker only when: edge > 2 * taker_fee (compensate for fees)
      - Size capped at $2K per leg to avoid book impact
    """

    MAKER_REBATE = 0.000     # Polymarket maker rebate (as of 2026: 0%)
    TAKER_FEE = 0.0175      # Polymarket taker fee (midpoint estimate)

    def __init__(self, api_key: str = "", secret: str = ""):
        self.api_key = api_key
        self.secret = secret

    def should_taker(self, edge_cents: float) -> bool:
        """Only take (pay fee) when edge sufficiently exceeds taker fee cost."""
        taker_cost_cents = self.TAKER_FEE * 100  # ~1.75 cents per dollar
        # Both legs pay taker fee on entry
        total_taker_cost = 2 * taker_cost_cents
        return edge_cents > total_taker_cost + 0.5

    async def execute_arb(self, opp: ArbOpportunity) -> dict:
        """
        Execute the dislocation arb. Returns execution report.

        NOTE: This is a skeleton. Real implementation requires:
          - Async REST/WebSocket order placement on Polymarket CLOB
          - Conditional token redemption on Polygon
          - Position tracking and partial-fill handling
        """
        size = min(opp.max_leg_size, 2000)  # cap at $2K

        strategy = "MAKER" if not self.should_taker(opp.edge_cents) else "TAKER"

        print(f"[EXEC] ✦ Dislocation Arb Detected — {opp.market_id}")
        print(f"[EXEC]   Combined cost: ${opp.combined_cost:.4f} | Edge: {opp.edge_cents:.2f}c")
        print(f"[EXEC]   Fee-adj edge: {opp.fee_adjusted_edge:.2f}c | Strategy: {strategy}")
        print(f"[EXEC]   Leg size: ${size:.2f} | Max book depth: ${opp.max_leg_size:.2f}")
        print(f"[EXEC]   → BUY YES @ ${opp.max_leg_size * 0.505:.4f} | BUY NO @ ${opp.max_leg_size * 0.495:.4f}")
        print(f"[EXEC]   → Guaranteed payout: ${size:.2f} | Net profit: ${opp.fee_adjusted_edge * size / 100:.4f}")

        return {
            "status": "dry_run",
            "market_id": opp.market_id,
            "edge_cents": opp.edge_cents,
            "fee_adj_edge_cents": opp.fee_adjusted_edge,
            "leg_size": size,
            "strategy": strategy,
        }


# ------------------------------------------------------------------------------
# Demo / Backtest Entry Point
# ------------------------------------------------------------------------------

async def demo():
    """Demonstrate the engine with mock data."""
    market_ids = ["btc-5min-up", "eth-5min-up", "sol-5min-up"]
    source = MockPolymarketWS(market_ids, seed=99)
    engine = ShortDurationArbEngine(
        min_edge_cents=0.50,
        max_position_usd=2000,
        fee_rate=0.0175,
    )
    router = ExecutionRouter()

    def on_opportunity(opp: ArbOpportunity):
        asyncio.create_task(router.execute_arb(opp))

    engine.add_callback(on_opportunity)
    await engine.run(market_ids, source)


if __name__ == "__main__":
    print("=" * 60)
    print("Short-Duration Price Dislocation Arbitrage — Reference Impl")
    print("=" * 60)
    print()
    print("DEMO: running 60-second backtest with mock Polymarket data")
    print("Injecting ~1% dislocation probability per tick (50ms interval)")
    print()

    async def timed_demo():
        try:
            await asyncio.wait_for(demo(), timeout=60.0)
        except asyncio.TimeoutError:
            print("\n[Demo] 60s elapsed — stopping.")

    asyncio.run(timed_demo())
