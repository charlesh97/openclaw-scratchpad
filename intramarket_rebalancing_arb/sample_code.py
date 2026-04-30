"""
Intra-Market Rebalancing Arbitrage (IMRA) — Reference Implementation
======================================================================
Detects YES/NO price-sum violations within single-condition Polymarket markets.

In a binary market: YES_price + NO_price should == $1.00 at all times.
When YES_ask + NO_ask < $1.00: buy both legs → lock in guaranteed spread.
When YES_bid + NO_bid > $1.00: sell both legs → reverse arb.

Note: This is a DETECTION + SIMULATION engine. Actual execution requires:
  - API credentials with order-posting permissions
  - Server co-location or low-latency connectivity
  - Real-time fill tracking

Author: vega research
Source: arXiv:2508.03474 (IMDEA Networks, 2025)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional
import logging

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConditionOrderBook:
    """Order book snapshot for a single YES/NO condition pair."""
    condition_id: str
    question: str
    timestamp: float

    yes_bid: float       # highest bid for YES
    yes_ask: float       # lowest ask for YES
    yes_bid_size: float  # size at best bid
    yes_ask_size: float  # size at best ask

    no_bid: float
    no_ask: float
    no_bid_size: float
    no_ask_size: float

    # Computed
    long_spread: float = field(init=False)   # YES_ask + NO_ask - 1.0
    short_spread: float = field(init=False)  # YES_bid + NO_bid - 1.0
    mid_spread: float = field(init=False)    # (YES_ask + NO_ask + YES_bid + NO_bid) / 2 - 1.0

    def __post_init__(self):
        self.long_spread = (self.yes_ask + self.no_ask) - 1.0
        self.short_spread = (self.yes_bid + self.no_bid) - 1.0
        mid_yes = (self.yes_bid + self.yes_ask) / 2
        mid_no = (self.no_bid + self.no_ask) / 2
        self.mid_spread = (mid_yes + mid_no) - 1.0

    def has_long_opportunity(self, min_spread: float = 0.005) -> bool:
        """Returns True if buying both legs locks in a profit."""
        return self.long_spread < -min_spread  # negative = profit

    def has_short_opportunity(self, min_spread: float = 0.005) -> bool:
        """Returns True if selling both legs locks in a profit."""
        return self.short_spread > min_spread

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "question": self.question,
            "timestamp": self.timestamp,
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "no_bid": self.no_bid,
            "no_ask": self.no_ask,
            "long_spread": self.long_spread,
            "short_spread": self.short_spread,
            "mid_spread": self.mid_spread,
            "long_opportunity": self.has_long_opportunity(),
            "short_opportunity": self.has_short_opportunity(),
        }


@dataclass
class ArbOpportunity:
    """A detected, actionable arbitrage opportunity."""
    orderbook: ConditionOrderBook
    opportunity_type: str          # "LONG" (buy both) or "SHORT" (sell both)
    grossSpread: float             # absolute spread in dollars
    gross_pct: float               # spread as % of notional ($1.00)
    leg_a: dict                    # {side, asset, price, size}
    leg_b: dict
    detected_at: float = field(default_factory=time.time)
    ttl_seconds: float = 2.7       # median window per IMDEA research

    def is_expired(self) -> bool:
        return (time.time() - self.detected_at) > self.ttl_seconds

    def expected_value(self) -> float:
        """Assuming resolution at 50/50 (worst case), EV of $1 deployed."""
        # LONG: pays $1 for YES + $1 for NO, receives $2 at resolution, cost = sum of asks
        # EV per $1 deployed = gross_profit / total_cost
        if self.opportunity_type == "LONG":
            cost = self.leg_a["price"] + self.leg_b["price"]
            return (1.0 - cost) / cost
        return (self.grossSpread - 0) / 1.0  # SHORT


# ---------------------------------------------------------------------------
# Polymarket API client (read-only — market data + orderbook)
# ---------------------------------------------------------------------------

POLYMARKET_GRAPHQL = "https://graphql.polymarket.com/graphql"

# Simplified market info + orderbook query
MARKET_QUERY = """
query GetMarkets($conditionId: String!) {
  market(conditionId: $conditionId) {
    conditionId
    question
    outcomes
    acceptance
    endDate_iso
    volume
    liquidity
    __typename
  }
}
"""

ORDERBOOK_QUERY = """
query GetOrderBook($conditionId: String!) {
  orderBook(conditionId: $conditionId) {
    bids {
      price
      size
    }
    asks {
      price
      size
    }
  }
}
"""

# We also need a way to list active single-condition markets
ACTIVE_MARKETS_QUERY = """
query GetActiveMarkets($limit: Int!) {
  markets(
    filter: { archive: false, closed: false }
    limit: $limit
  ) {
    conditionId
    question
    outcomes
    endDate_iso
    volume
    liquidity
    __typename
  }
}
"""


class PolymarketClient:
    """Minimal read-only client for Polymarket market + orderbook data."""

    def __init__(self, session=None):
        self._session = session or requests_session()

    async def get_active_markets(self, limit: int = 500) -> list[dict]:
        """Fetch recently active markets (candidates for arb scanning)."""
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                POLYMARKET_GRAPHQL,
                json={"query": ACTIVE_MARKETS_QUERY, "variables": {"limit": limit}},
                headers={"Content-Type": "application/json"},
                timeout=10.0,
            )
        data = resp.json()
        return data.get("data", {}).get("markets", [])

    async def get_orderbook(self, condition_id: str) -> Optional[dict]:
        """Fetch orderbook for a single condition."""
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                POLYMARKET_GRAPHQL,
                json={"query": ORDERBOOK_QUERY, "variables": {"conditionId": condition_id}},
                headers={"Content-Type": "application/json"},
                timeout=10.0,
            )
        data = resp.json()
        return data.get("data", {}).get("orderBook")

    def build_orderbook_snapshot(
        self, market: dict, orderbook: dict
    ) -> Optional[ConditionOrderBook]:
        """
        Extract best bid/ask for YES and NO from an orderbook response.
        Polymarket orderbook entries are indexed by outcome (e.g., "YES", "NO").
        """
        bids = orderbook.get("bids", []) or []
        asks = orderbook.get("asks", []) or []

        if not bids or not asks:
            return None

        # Best bid/ask
        yes_bid = yes_ask = no_bid = no_ask = None
        yes_bid_size = yes_ask_size = no_bid_size = no_ask_size = 0.0

        for entry in bids:
            outcome = entry.get("outcome", "").upper()
            price = float(entry.get("price", 0))
            size = float(entry.get("size", 0))
            if outcome == "YES" and (yes_bid is None or price > yes_bid):
                yes_bid, yes_bid_size = price, size
            elif outcome == "NO" and (no_bid is None or price > no_bid):
                no_bid, no_bid_size = price, size

        for entry in asks:
            outcome = entry.get("outcome", "").upper()
            price = float(entry.get("price", 0))
            size = float(entry.get("size", 0))
            if outcome == "YES" and (yes_ask is None or price < yes_ask):
                yes_ask, yes_ask_size = price, size
            elif outcome == "NO" and (no_ask is None or price < no_ask):
                no_ask, no_ask_size = price, size

        if None in (yes_bid, yes_ask, no_bid, no_ask):
            return None

        return ConditionOrderBook(
            condition_id=market["conditionId"],
            question=market["question"],
            timestamp=time.time(),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            yes_bid_size=yes_bid_size,
            yes_ask_size=yes_ask_size,
            no_bid=no_bid,
            no_ask=no_ask,
            no_bid_size=no_bid_size,
            no_ask_size=no_ask_size,
        )


# ---------------------------------------------------------------------------
# Core detection engine
# ---------------------------------------------------------------------------

class IMRADetector:
    """
    Intra-Market Rebalancing Arbitrage detector.

    Scans single-condition Polymarket markets for YES/NO price-sum violations.
    Suitable for running as a continuous scan loop or periodic batch.
    """

    def __init__(
        self,
        min_spread_bps: float = 50.0,    # min spread in basis points (50 bps = 0.5%)
        min_liquidity_usd: float = 100.0, # min combined orderbook depth
        max_age_seconds: float = 5.0,     # stale opportunity discard
        client: Optional[PolymarketClient] = None,
    ):
        self.min_spread_bps = min_spread_bps
        self.min_spread = min_spread_bps / 10_000
        self.min_liquidity = min_liquidity_usd
        self.max_age = max_age_seconds
        self.client = client or PolymarketClient()
        self.logger = logging.getLogger("IMRA")

        # State
        self.opportunities: list[ArbOpportunity] = []
        self.scan_count = 0
        self.detection_count = 0

    async def scan_markets(self, market_ids: list[str]) -> list[ArbOpportunity]:
        """
        Scan a batch of markets for rebalancing arb opportunities.
        Returns list of detected opportunities sorted by spread size (largest first).
        """
        opportunities = []
        self.scan_count += len(market_ids)

        for market in market_ids:
            orderbook = await self.client.get_orderbook(market["conditionId"])
            if not orderbook:
                continue

            snapshot = self.client.build_orderbook_snapshot(market, orderbook)
            if not snapshot:
                continue

            # Filter by liquidity
            total_depth = (
                snapshot.yes_bid * snapshot.yes_bid_size
                + snapshot.yes_ask * snapshot.yes_ask_size
                + snapshot.no_bid * snapshot.no_bid_size
                + snapshot.no_ask * snapshot.no_ask_size
            )
            if total_depth < self.min_liquidity:
                continue

            # Detect LONG opportunity (buy both legs)
            if snapshot.has_long_opportunity(min_spread=self.min_spread):
                opp = ArbOpportunity(
                    orderbook=snapshot,
                    opportunity_type="LONG",
                    grossSpread=abs(snapshot.long_spread),
                    gross_pct=abs(snapshot.long_spread) * 100,
                    leg_a={
                        "side": "BUY",
                        "asset": "YES",
                        "price": snapshot.yes_ask,
                        "size": min(snapshot.yes_ask_size, snapshot.no_ask_size),
                    },
                    leg_b={
                        "side": "BUY",
                        "asset": "NO",
                        "price": snapshot.no_ask,
                        "size": min(snapshot.yes_ask_size, snapshot.no_ask_size),
                    },
                )
                opportunities.append(opp)
                self.detection_count += 1
                self.logger.info(
                    f"LONG opp: {snapshot.question[:60]} | spread={opp.gross_pct:.2f}% | "
                    f"YES_ask={snapshot.yes_ask:.4f} NO_ask={snapshot.no_ask:.4f}"
                )

            # Detect SHORT opportunity (sell both legs)
            elif snapshot.has_short_opportunity(min_spread=self.min_spread):
                opp = ArbOpportunity(
                    orderbook=snapshot,
                    opportunity_type="SHORT",
                    grossSpread=abs(snapshot.short_spread),
                    gross_pct=abs(snapshot.short_spread) * 100,
                    leg_a={
                        "side": "SELL",
                        "asset": "YES",
                        "price": snapshot.yes_bid,
                        "size": min(snapshot.yes_bid_size, snapshot.no_bid_size),
                    },
                    leg_b={
                        "side": "SELL",
                        "asset": "NO",
                        "price": snapshot.no_bid,
                        "size": min(snapshot.yes_bid_size, snapshot.no_bid_size),
                    },
                )
                opportunities.append(opp)
                self.detection_count += 1

        self.opportunities = sorted(opportunities, key=lambda x: x.grossSpread, reverse=True)
        return self.opportunities

    async def run_scan_loop(self, interval_seconds: float = 1.0):
        """
        Continuous scan loop. Yields opportunities as they are detected.
        Designed for use with `async for` or as a background task.
        """
        self.logger.info("Starting IMRA continuous scan loop")
        while True:
            try:
                markets = await self.client.get_active_markets(limit=200)
                await self.scan_markets(markets)

                if self.opportunities:
                    yield self.opportunities

            except Exception as e:
                self.logger.error(f"Scan error: {e}")

            await asyncio.sleep(interval_seconds)

    def stats(self) -> dict:
        return {
            "total_scanned": self.scan_count,
            "opportunities_detected": self.detection_count,
            "active_opportunities": len(self.opportunities),
            "min_spread_bps": self.min_spread_bps,
            "min_liquidity_usd": self.min_liquidity,
        }


# ---------------------------------------------------------------------------
# Execution layer (stub — requires API credentials + co-location)
# ---------------------------------------------------------------------------

class ExecutionSimulator:
    """
    Simulates order execution for backtesting and paper trading.
    Real execution would use Polymarket order API with signed requests.
    """

    def __init__(self, fill_rate: float = 0.3):
        """
        fill_rate: fraction of posted orders that fill (calibrate from real data)
        """
        self.fill_rate = fill_rate
        self.filled = []
        self.missed = []

    def simulate_execution(self, opportunity: ArbOpportunity) -> dict:
        import random
        filled = random.random() < self.fill_rate
        result = {
            "opportunity": opportunity,
            "filled": filled,
            "executed_at": time.time(),
        }
        if filled:
            self.filled.append(result)
        else:
            self.missed.append(result)
        return result

    def paper_pnl(self) -> dict:
        if not self.filled:
            return {"pnl": 0.0, "realized_spreads": 0}

        total_pnl = sum(
            (1.0 - opp.orderbook.yes_ask - opp.orderbook.no_ask)
            for r in self.filled
            if r["opportunity"].opportunity_type == "LONG"
        )
        return {
            "pnl": total_pnl,
            "filled_count": len(self.filled),
            "missed_count": len(self.missed),
            "fill_rate_observed": len(self.filled) / (len(self.filled) + len(self.missed)),
        }


# ---------------------------------------------------------------------------
# CLI demo (paper trade / backtest simulation)
# ---------------------------------------------------------------------------

async def demo():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("IMRA-demo")

    client = PolymarketClient()
    detector = IMRADetector(min_spread_bps=50.0, min_liquidity_usd=200.0)
    simulator = ExecutionSimulator(fill_rate=0.3)

    logger.info("Fetching active markets...")
    markets = await client.get_active_markets(limit=100)

    logger.info(f"Scanning {len(markets)} markets...")
    opps = await detector.scan_markets(markets)

    if not opps:
        logger.info("No opportunities found above threshold.")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"DETECTED {len(opps)} OPPORTUNITIES")
    logger.info(f"{'='*60}")

    for opp in opps[:10]:
        snapshot = opp.orderbook
        print(f"\n  Q: {snapshot.question[:80]}")
        print(f"  Type: {opp.opportunity_type} | Spread: {opp.gross_pct:.3f}%")
        print(f"  YES ask={snapshot.yes_ask:.4f}  NO ask={snapshot.no_ask:.4f}")
        print(f"  YES bid={snapshot.yes_bid:.4f}  NO bid={snapshot.no_bid:.4f}")

        # Paper trade simulation
        result = simulator.simulate_execution(opp)
        if result["filled"]:
            print(f"  [FILLED] → gross profit = ${opp.grossSpread:.4f}")
        else:
            print(f"  [MISSED] → fill not received")

    pnl = simulator.paper_pnl()
    print(f"\n{'='*60}")
    print(f"SIMULATION SUMMARY:")
    print(f"  Filled:   {pnl['filled_count']}")
    print(f"  Missed:   {pnl['missed_count']}")
    print(f"  Observed fill rate: {pnl['fill_rate_observed']:.1%}")
    print(f"  Paper P&L: ${pnl['pnl']:.4f}")
    print(f"\n  Detector stats: {detector.stats()}")


if __name__ == "__main__":
    # NOTE: Requires network access to Polymarket GraphQL API
    # For a real deployment, add API credentials and server co-location
    try:
        asyncio.run(demo())
    except Exception as e:
        print(f"Demo error (expected if no network): {e}")
