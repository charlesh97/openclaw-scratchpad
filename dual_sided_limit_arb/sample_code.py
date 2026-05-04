#!/usr/bin/env python3
"""
Dual-Sided Limit Arbitrage with Maker Optimization — Reference Implementation
==============================================================================
Places maker limit orders on BOTH sides of a binary prediction market simultaneously.
Profits when: maker rebate + convergence spread > opportunity cost.

The core insight: Polymarket's taker fee (1.75%) is high relative to maker rebate (~0%).
By placing limit orders on both sides, you:
  1. Avoid paying the taker fee (zero cost to post)
  2. Earn a small spread when the market converges
  3. Remain market-neutral once both legs fill

If only one leg fills, you must hedge or exit — partial fill = directional risk.

This is a REFERENCE implementation for educational purposes.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ------------------------------------------------------------------------------
# Constants & Fee Model
# ------------------------------------------------------------------------------

class Platform(Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


@dataclass
class PlatformFees:
    maker_fee: float   # what you pay to post a maker order (often 0)
    taker_fee: float  # what you pay to take liquidity immediately
    rebate: float     # rebate for providing liquidity (can be negative = cost)


FEES = {
    Platform.POLYMARKET: PlatformFees(
        maker_fee=0.000,   # ~0% to post a limit order
        taker_fee=0.0175,  # ~1.75% taker fee (midpoint estimate)
        rebate=0.000,       # no maker rebate as of 2026
    ),
    Platform.KALSHI: PlatformFees(
        maker_fee=0.001,    # ~0.1% maker fee
        taker_fee=0.0200,   # ~2.0% taker fee
        rebate=0.000,
    ),
}

# Per-leg settlement fee (fees paid when contract resolves)
SETTLEMENT_FEE = 0.000  # no settlement fee in this model


# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------

@dataclass
class MarketQuote:
    """A snapshot of a binary market's current price state."""
    market_id: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    timestamp: float


@dataclass
class DualMakerOrder:
    """A pair of maker orders placed on opposite sides of a market."""
    market_id: str
    yes_order_id: Optional[str] = None
    no_order_id: Optional[str] = None
    yes_price: float = 0.0   # limit order price for YES
    no_price: float = 0.0   # limit order price for NO
    yes_filled: bool = False
    no_filled: bool = False
    placed_at: float = 0.0


@dataclass
class FillEvent:
    """Confirmed fill on one leg of a dual-maker strategy."""
    market_id: str
    side: str           # "YES" or "NO"
    fill_price: float
    fill_size: float    # notional in USDC
    order_id: str
    remaining_lag_ms: float
    timestamp: float


# ------------------------------------------------------------------------------
# Core Arbitrage Logic
# ------------------------------------------------------------------------------

def compute_dual_maker_edge(
    quote: MarketQuote,
    platform: Platform = Platform.POLYMARKET,
    maker_spread_cents: float = 0.5,
) -> dict:
    """
    Compute whether a dual-maker placement is profitable.

    Strategy:
      - Post YES buy limit at: quote.yes_bid + epsilon (slightly better than current bid)
      - Post NO buy limit at: quote.no_bid + epsilon

    The epsilon (maker_spread_cents) ensures your orders are first in queue
    and earn the spread between bid and ask.

    Profit from dual-maker comes from:
      1. Avoiding taker fee (savings: ~1.75% on Polymarket)
      2. Earning the bid-ask spread when both legs fill and market converges

    Simple model: if both legs fill, net cost = (yes_bid + no_bid) + maker_fee
    Payout at settlement = $1.00 per leg

    gross_profit_per_dollar = 1.00 - (yes_bid + no_bid)
    maker_cost              = (yes_bid + no_bid) * maker_fee_rate
    net_profit_per_dollar   = gross_profit - maker_cost

    Returns dict with edge analysis or None if not profitable.
    """
    fees = FEES[platform]

    # Combined cost at bid (what you pay to buy both sides from the order book)
    combined_bid_cost = quote.yes_bid + quote.no_bid

    # Gross edge when both legs filled at bid and settled: $1 received per leg
    gross_edge = 1.00 - combined_bid_cost

    # Maker cost (post both legs at bid)
    maker_cost = combined_bid_cost * fees.maker_fee

    # Avoided taker fee value: what we'd pay if we took instead
    avoided_taker = combined_bid_cost * fees.taker_fee

    # Net edge = gross edge + avoided taker fee - maker cost
    # This is the structural advantage of being a maker vs taker
    net_edge = gross_edge + avoided_taker - maker_cost

    # Max leg size (limited by order book depth on each side)
    max_size_per_leg = 5000  # approximate from $5K–$15K book depth cited in Coindesk

    # Place orders slightly inside the spread
    yes_limit_price = quote.yes_bid + (maker_spread_cents / 100)
    no_limit_price = quote.no_bid + (maker_spread_cents / 100)

    return {
        "market_id": quote.market_id,
        "combined_bid_cost": combined_bid_cost,
        "gross_edge": gross_edge,
        "avoided_taker": avoided_taker,
        "net_edge": net_edge,
        "yes_limit_price": yes_limit_price,
        "no_limit_price": no_limit_price,
        "max_size_per_leg": max_size_per_leg,
        "viable": net_edge > 0.005,  # need at least 0.5 cents edge after fees
        "platform": platform.value,
    }


# ------------------------------------------------------------------------------
# Order Manager (skeleton)
# ------------------------------------------------------------------------------

class DualMakerOrderManager:
    """
    Manages dual-sided limit order placement and fill tracking.

    Core state machine:
      1. SCANNING: monitoring markets for viable dual-maker opportunities
      2. PENDING: both legs placed, waiting for fills
      3. PARTIAL_FILL: one leg filled, monitoring for hedge
      4. FILLED: both legs filled, market-neutral position locked
      5. EXPIRED: orders cancelled after timeout, capital released

    Replacement for: arb-bot-main's aggressive market-order YOLO approach.
    """

    def __init__(
        self,
        min_edge_cents: float = 0.30,
        max_position_usd: float = 3000,
        order_timeout_seconds: float = 240,  # 4 min max for 5-min markets
        partial_fill_window_ms: float = 5000,
        platform: Platform = Platform.POLYMARKET,
    ):
        self.min_edge_cents = min_edge_cents
        self.max_position_usd = max_position_usd
        self.order_timeout_seconds = order_timeout_seconds
        self.partial_fill_window_ms = partial_fill_window_ms
        self.platform = platform
        self.active_orders: dict[str, DualMakerOrder] = {}
        self.partial_fills: dict[str, FillEvent] = {}  # market_id -> first fill

    def should_place(self, edge_analysis: dict) -> bool:
        """Check if a dual-maker placement is worth the capital commitment."""
        if not edge_analysis["viable"]:
            return False
        if edge_analysis["net_edge"] < self.min_edge_cents / 100:
            return False
        return True

    async def place_dual_maker(
        self,
        market_id: str,
        yes_price: float,
        no_price: float,
        size: float,
        api_client,
    ) -> DualMakerOrder:
        """
        Place a dual-maker order on both sides of a binary market.

        NOTE: This is a skeleton. Real implementation requires:
          - POST /orders to Polymarket CLOB (via pmxt or Gamma API)
          - WebSocket subscription for fill confirmations
          - Cancellation on partial fill or timeout

        Args:
            market_id: target market identifier
            yes_price: limit price for YES leg
            no_price: limit price for NO leg
            size: notional size per leg in USDC
            api_client: real API client with .place_limit_order() method
        """
        order = DualMakerOrder(
            market_id=market_id,
            yes_price=yes_price,
            no_price=no_price,
            placed_at=time.time(),
        )

        print(f"[DualMaker] → Placing BUY YES @ ${yes_price:.4f} | BUY NO @ ${no_price:.4f}")
        print(f"[DualMaker]   Size: ${size:.2f}/leg | Timeout: {self.order_timeout_seconds}s")

        # --- REAL API CALLS (replace mock with actual) ---
        # yes_result = await api_client.place_limit_order(market_id, "YES", "BUY", size, yes_price)
        # no_result  = await api_client.place_limit_order(market_id, "NO",  "BUY", size, no_price)
        # order.yes_order_id = yes_result["order_id"]
        # order.no_order_id = no_result["order_id"]
        # -----------------------------------------------

        # Mock placeholders
        order.yes_order_id = f"MOCK-{market_id}-YES-{int(time.time())}"
        order.no_order_id  = f"MOCK-{market_id}-NO-{int(time.time())}"
        self.active_orders[market_id] = order

        return order

    async def on_fill(self, fill: FillEvent):
        """
        Called when a fill event is received from the WebSocket feed.
        Updates order state machine and handles partial-fill scenarios.
        """
        order = self.active_orders.get(fill.market_id)
        if not order:
            return

        if fill.side == "YES":
            order.yes_filled = True
            order.yes_order_id = fill.order_id  # confirmed
        else:
            order.no_filled = True
            order.no_order_id = fill.order_id

        if order.yes_filled and order.no_filled:
            print(f"[DualMaker] ✓ BOTH LEGS FILLED — {fill.market_id} | Market neutral position confirmed")
            del self.active_orders[fill.market_id]
            return

        # Partial fill — record and log warning
        self.partial_fills[fill.market_id] = fill
        remaining_side = "NO" if fill.side == "YES" else "YES"
        print(f"[DualMaker] ⚠ PARTIAL FILL — {fill.market_id} | {fill.side} filled @ ${fill.fill_price:.4f}")
        print(f"[DualMaker]   → {remaining_side} still open, monitoring for {self.partial_fill_window_ms}ms")
        print(f"[DualMaker]   → Consider hedging via opposite platform or closing at market")

        # Set up a timeout to close remaining leg if not filled
        asyncio.get_event_loop().create_task(
            self._check_partial_fill_timeout(fill.market_id, remaining_side)
        )

    async def _check_partial_fill_timeout(self, market_id: str, remaining_side: str):
        """If remaining leg not filled within window, close the position."""
        await asyncio.sleep(self.partial_fill_window_ms / 1000)
        order = self.active_orders.get(market_id)
        if order and not (order.yes_filled if remaining_side == "YES" else order.no_filled):
            print(f"[DualMaker] ⚠ TIMEOUT — {market_id} {remaining_side} not filled, cancelling remaining leg")
            # await api_client.cancel_order(market_id, remaining_side)
            del self.active_orders[market_id]


# ------------------------------------------------------------------------------
# Mock Data Source (for backtesting / development)
# ------------------------------------------------------------------------------

class MockPolymarketCLOB:
    """
    Mock Polymarket CLOB API for backtesting dual-maker strategy.
    Replace with real Gamma API / Polymarket CLOB client.

    Generates realistic bid-ask spreads for short-duration binary markets.
    """

    SPREAD_BPS = 50  # 0.50% spread (50 basis points = typical for short-duration markets)

    def __init__(self, market_ids: list[str], seed: int = 123):
        import random
        self.market_ids = market_ids
        self.rng = random.Random(seed)
        self.fair_prices = {mid: 0.50 for mid in market_ids}
        self.half_spread = self.SPREAD_BPS / 10000 / 2  # 0.25% each side

    async def stream_quotes(self, market_id: str):
        """Stream simulated order book quotes at ~5Hz."""
        while True:
            fair = self.fair_prices.get(market_id, 0.50)
            # Random walk of fair price (simulates market drift)
            drift = self.rng.gauss(0, 0.002)
            self.fair_prices[market_id] = max(0.01, min(0.99, fair + drift))
            fair = self.fair_prices[market_id]

            half = self.half_spread
            quote = MarketQuote(
                market_id=market_id,
                yes_bid=round(fair - half, 4),
                yes_ask=round(fair + half, 4),
                no_bid=round((1 - fair) - half, 4),
                no_ask=round((1 - fair) + half, 4),
                timestamp=time.time(),
            )
            yield quote
            await asyncio.sleep(0.2)  # 5Hz


# ------------------------------------------------------------------------------
# Strategy Monitor
# ------------------------------------------------------------------------------

class DualMakerStrategy:
    """
    Top-level strategy orchestrator for dual-sided limit arbitrage.

    Workflow:
      1. Scan markets for quotes
      2. Compute dual-maker edge for each
      3. Place dual-maker orders when viable
      4. Track fills and handle partial fills
    """

    def __init__(
        self,
        manager: DualMakerOrderManager,
        min_edge_cents: float = 0.30,
        max_size_per_leg: float = 3000,
        scan_interval_seconds: float = 5,
    ):
        self.manager = manager
        self.min_edge_cents = min_edge_cents
        self.max_size_per_leg = max_size_per_leg
        self.scan_interval_seconds = scan_interval_seconds
        self.placed_count = 0
        self.both_filled_count = 0
        self.partial_fill_count = 0

    async def scan_and_place(self, market_id: str, quote: MarketQuote, api_client):
        """Evaluate a market quote and place dual-maker if viable."""
        edge = compute_dual_maker_edge(quote, Platform.POLYMARKET)
        if not self.manager.should_place(edge):
            return

        size = min(self.max_size_per_leg, edge["max_size_per_leg"])
        await self.manager.place_dual_maker(
            market_id=market_id,
            yes_price=edge["yes_limit_price"],
            no_price=edge["no_limit_price"],
            size=size,
            api_client=api_client,
        )
        self.placed_count += 1
        print(f"[Strategy] Placed #{self.placed_count} | Net edge: {edge['net_edge']*100:.2f}c | "
              f"Combined cost: ${edge['combined_bid_cost']:.4f}")

    async def run(self, market_ids: list[str], data_source, api_client):
        """Main scanning and placement loop."""
        print(f"[DualMakerStrategy] Started — scanning {len(market_ids)} markets every {self.scan_interval_seconds}s")
        print(f"[DualMakerStrategy] Min edge: {self.min_edge_cents}c | Max size: ${self.max_size_per_leg}/leg")
        print()

        async def scan_loop(market_id: str):
            async for quote in data_source.stream_quotes(market_id):
                try:
                    await self.scan_and_place(market_id, quote, api_client)
                except Exception as e:
                    print(f"[Strategy] Error scanning {market_id}: {e}")

        await asyncio.gather(*[scan_loop(mid) for mid in market_ids])


# ------------------------------------------------------------------------------
# Demo / Backtest Entry Point
# ------------------------------------------------------------------------------

async def demo():
    """Run a 60-second backtest with mock data."""
    market_ids = ["btc-5min-up", "eth-5min-up", "sol-5min-up"]
    data_source = MockPolymarketCLOB(market_ids, seed=42)
    manager = DualMakerOrderManager(
        min_edge_cents=0.30,
        max_position_usd=3000,
    )
    strategy = DualMakerStrategy(
        manager=manager,
        min_edge_cents=0.30,
        max_size_per_leg=3000,
    )

    print("=" * 60)
    print("Dual-Sided Limit Arbitrage — 60s Backtest")
    print("=" * 60)
    print()

    # Mock API client (replace with real Polymarket CLOB client in production)
    mock_api = type("MockAPI", (), {})()

    try:
        await asyncio.wait_for(strategy.run(market_ids, data_source, mock_api), timeout=60.0)
    except asyncio.TimeoutError:
        print("\n[Demo] 60s elapsed.")
        print(f"[Demo] Summary: {strategy.placed_count} dual-maker orders placed")


if __name__ == "__main__":
    asyncio.run(demo())
