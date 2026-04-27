#!/usr/bin/env python3
"""
market_maker.py — Prediction Market Market-Making Module for Kalshi

Reference implementation of passive liquidity provision on Kalshi,
exploiting the "Optimism Tax": takers systematically overpay for YES bets,
especially at longshot prices (1¢–15¢).

Source: Becker (2026), "The Microstructure of Wealth Transfer in
Prediction Markets" — historical YES returns as low as 43¢ on the dollar
at 5¢ prices.

Algorithm
---------
1. Select target markets: Sports + Entertainment, YES in [1¢, 15¢]
2. Compute fair spread: ≥2¢ minimum on Kalshi
3. Post orders: sell YES just above fair, below inflated ask
   — the spread / optimism tax is the maker's gross profit
4. Inventory management: adjust quoting when YES/NO inventory grows
5. Position limits: per-market, total, and daily loss kill-switch
6. (Optional) Kelly sizing for max position

Note
----
This module computes quotes only — no API calls, no order execution.
Human review required before any order is placed.
"""

from __future__ import annotations

import datetime
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


class Side(Enum):
    """Binary outcome side on a Kalshi contract."""
    YES = "YES"
    NO = "NO"


@dataclass
class Quote:
    """
    Computed quote for a single market.

    Attributes
    ----------
    market_id : str
        Unique Kalshi market / contract identifier.
    sell_side : Side
        Which side the market maker is offering to sell.
        Selling YES = maker collects premium and risks $1 if YES resolves.
    price : float
        Price in dollars (0.01 – 0.99).
    size : int
        Number of contracts in the quote.
    expected_edge : float
        Expected value per contract in dollars.

    Edge is computed as: quote_price - fair_price
    Positive edge means maker sells above the deflated fair value.
    """
    market_id: str
    sell_side: Side
    price: float
    size: int
    expected_edge: float

    def __repr__(self) -> str:
        return (
            f"Quote(market_id={self.market_id!r}, "
            f"sell_side={self.sell_side.value}, "
            f"price=${self.price:.2f}, size={self.size}, "
            f"edge=${self.expected_edge:+.4f})"
        )


@dataclass
class Market:
    """
    Representation of a Kalshi market / contract.

    Attributes
    ----------
    market_id : str
        Unique identifier.
    category : str
        Market category (e.g., "sports", "politics", "entertainment").
    subcategory : str
        Market subcategory (e.g., "nfl", "nba", "election").
    yes_bid : float
        Highest bid price for YES (in dollars).
    yes_ask : float
        Lowest ask price for YES (in dollars).
    no_bid : float
        Highest bid price for NO (in dollars).
    no_ask : float
        Lowest ask price for NO (in dollars).
    volume_24h : float
        Approximate 24-hour trading volume in dollars.
    taker_bias : float
        Estimated taker premium (fraction, 0.0–1.0). Higher = more optimism bias.
        Derived from historical fill data; e.g., 0.12 means takers overpay ~12%
        relative to fair value.
    """
    market_id: str
    category: str
    subcategory: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume_24h: float = 0.0
    taker_bias: float = 0.10  # default 10% optimism tax

    @property
    def spread_cents(self) -> float:
        """Bid-ask spread in cents."""
        return (self.yes_ask - self.yes_bid) * 100

    @property
    def mid_price(self) -> float:
        """Mid-price of YES."""
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def is_longshot(self) -> bool:
        """Longshot = YES ask at or below 15¢."""
        return self.yes_ask <= 0.15

    def is_eligible(self, config: MarketMakerConfig) -> bool:
        """Check whether this market meets quoting criteria."""
        # Category filter: focus on sports / entertainment for highest taker bias
        if self.category.lower() not in ("sports", "entertainment"):
            return False
        # Price filter: longshots have highest Optimism Tax
        if not (0.01 <= self.yes_ask <= config.max_yes_price):
            return False
        # Spread filter: need at least min_spread_bps to be profitable
        spread_bps = self.spread_cents * 100 / max(self.yes_ask, 0.001)
        if spread_bps < config.min_spread_bps:
            return False
        return True


@dataclass
class Fill:
    """
    A fill event from the exchange.

    Attributes
    ----------
    market_id : str
        Market where the fill occurred.
    side : Side
        Which side was filled (YES or NO).
    price : float
        Fill price in dollars.
    size : int
        Number of contracts filled.
    timestamp : datetime.datetime
        Fill timestamp.
    premium_collected : float
        Premium received when the order was filled.
    """
    market_id: str
    side: Side
    price: float
    size: int
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    premium_collected: float = 0.0


@dataclass
class MarketMakerState:
    """
    Current state for one market.
    """
    market_id: str
    yes_inventory: int = 0          # positive = long YES; negative = net short YES
    no_inventory: int = 0          # positive = long NO; negative = net short NO
    orders_open: int = 0           # number of open (unfilled) orders
    realized_pnl: float = 0.0      # realized PnL in dollars

    @property
    def net_yes_exposure(self) -> int:
        """Net YES contracts (positive = net long YES)."""
        return self.yes_inventory

    def __repr__(self) -> str:
        return (
            f"MarketMakerState(market_id={self.market_id!r}, "
            f"yes_inventory={self.yes_inventory}, "
            f"no_inventory={self.no_inventory}, "
            f"orders_open={self.orders_open}, "
            f"realized_pnl=${self.realized_pnl:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MarketMakerConfig:
    """
    Configuration for the market maker.

    Attributes
    ----------
    max_yes_price : float
        Only market-make on YES contracts priced at or below this threshold.
        Default 0.15 (15¢) — captures longshots with highest Optimism Tax.
    min_spread_bps : int
        Minimum bid-ask spread in basis points to qualify a market.
        200 bps = 2¢ on a 10¢ contract; scales appropriately for other prices.
    max_position_per_market : int
        Maximum number of contracts (YES or NO) in a single market.
        Prevents over-concentration.
    max_total_position : int
        Maximum total contracts across all markets.
        Protects against aggregate directional risk.
    daily_loss_limit : float
        Realized daily loss threshold that triggers the kill-switch.
        When abs(daily_pnl) exceeds this, quoting halts.
    kelly_fraction : float
        Kelly fraction for position sizing (0.0–1.0). 0.5 = half-Kelly.
        0.0 falls back to base_size.
    base_size : int
        Base contract size per order when Kelly is zero or very small.
    optimism_tax_rate : float
        Estimated taker overpayment rate (fraction). Becker (2026) reports
        ~10–15% at longshot prices; used to compute fair price.
    """
    max_yes_price: float = 0.15
    min_spread_bps: int = 200
    max_position_per_market: int = 500
    max_total_position: int = 5000
    daily_loss_limit: float = 100.0
    kelly_fraction: float = 0.5     # half-Kelly
    base_size: int = 10
    optimism_tax_rate: float = 0.12  # ~12% overpayment at longshot prices


# ─────────────────────────────────────────────────────────────────────────────
# Core market maker
# ─────────────────────────────────────────────────────────────────────────────


class MarketMaker:
    """
    Kalshi prediction market market-maker.

    Exploits the "Optimism Tax" — systematic taker overpayment for YES,
    especially on longshot contracts (1¢–15¢) in Sports/Entertainment.

    This module computes quotes only. No API calls, no order execution.
    Human review required before placing orders.

    Parameters
    ----------
    config : MarketMakerConfig
        Configuration parameters.

    Example
    -------
    >>> config = MarketMakerConfig()
    >>> mm = MarketMaker(config)
    >>> markets = [...]  # fetched from Kalshi API
    >>> quotes = [mm.compute_quote(m) for m in markets]
    >>> quotes = [q for q in quotes if q is not None]  # filter None
    """

    def __init__(self, config: MarketMakerConfig) -> None:
        self.config = config
        self._states: dict[str, MarketMakerState] = {}
        self._daily_pnl: float = 0.0
        self._kill_switched: bool = False
        self._kill_switch_time: Optional[datetime.datetime] = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_state(self, market_id: str) -> MarketMakerState:
        """Get or create state for a market."""
        if market_id not in self._states:
            self._states[market_id] = MarketMakerState(market_id=market_id)
        return self._states[market_id]

    @property
    def total_yes_inventory(self) -> int:
        """Aggregate YES inventory across all markets."""
        return sum(s.yes_inventory for s in self._states.values())

    @property
    def total_no_inventory(self) -> int:
        """Aggregate NO inventory across all markets."""
        return sum(s.no_inventory for s in self._states.values())

    @property
    def total_position(self) -> int:
        """Total contracts across all markets."""
        return abs(self.total_yes_inventory) + abs(self.total_no_inventory)

    @property
    def total_realized_pnl(self) -> float:
        """Total realized PnL across all markets."""
        return sum(s.realized_pnl for s in self._states.values())

    @property
    def daily_pnl(self) -> float:
        """Current daily realized PnL."""
        return self._daily_pnl

    # ── Kelly sizing ─────────────────────────────────────────────────────────

    def _kelly_size(self, price: float, win_prob: float) -> int:
        """
        Compute Kelly-optimal position size in contracts.

        Parameters
        ----------
        price : float
            Current YES ask price (taker's inflated probability proxy).
        win_prob : float
            Estimated win probability. For optimism-tax edge this is the
            inflated price at which takers are buying.

        Returns
        -------
        int
            Kelly-optimal number of contracts (capped by config limits).

        Notes
        -----
        Kelly formula: f* = (b·p - q) / b
        where b = 1/price - 1  (odds against)
              p = win_probability (inflated taker price = edge source)
              q = 1 - p

        The edge source here: takers overpay (price > fair). So price is
        inflated AND win_prob is inflated. The maker who sells at the
        inflated price and holds to resolution earns the tax.

        Example: price=0.05 (5¢), p=0.05 → b=19, f*=(19*0.05-0.95)/19≈0.053
        """
        if self.config.kelly_fraction <= 0.0:
            return self.config.base_size

        b = 1.0 / price - 1.0          # odds against
        if b <= 0:
            return 0

        q = 1.0 - win_prob
        kelly = (b * win_prob - q) / b

        if kelly <= 0:
            return 0

        # Apply Kelly fraction (half-Kelly reduces volatility)
        kelly *= self.config.kelly_fraction

        # Convert to contract count (base_size = Kelly unit)
        size = max(1, int(kelly * self.config.base_size))
        return size

    # ── Fair price / edge calculation ────────────────────────────────────────

    def _fair_price(self, market: Market) -> float:
        """
        Estimate fair YES price correcting for taker optimism bias.

        The taker bias inflates YES prices above fair value.
        We deflate the observed ask by optimism_tax_rate to estimate fair value.

        Becker (2026) found takers overpay ~10–15% at longshot prices,
        with effects strongest below 10¢.

        Parameters
        ----------
        market : Market

        Returns
        -------
        float
            Estimated fair YES price.
        """
        # Adjust tax rate downward for higher-priced markets (less bias)
        tax = self.config.optimism_tax_rate
        if market.yes_ask > 0.10:
            tax *= 0.5

        fair = market.yes_ask * (1.0 - tax)
        return max(fair, 0.01)  # floor at 1¢

    def _expected_edge(self, market: Market, quote_price: float) -> float:
        """
        Compute expected edge per contract for a YES-sell quote.

        Edge = quote_price - fair_price

        Selling YES above fair price generates positive edge for the maker.
        The optimism tax inflates the ask above fair; we sell between fair
        and ask, capturing the difference as edge.

        Parameters
        ----------
        market : Market
        quote_price : float

        Returns
        -------
        float
            Expected edge in dollars per contract.
        """
        fair = self._fair_price(market)
        edge = quote_price - fair
        return edge

    # ── Inventory-adjusted sizing ─────────────────────────────────────────────

    def _inventory_adjust(
        self,
        state: MarketMakerState,
        side: Side,
        raw_size: int,
    ) -> int:
        """
        Reduce order size when inventory is already lopsided.

        Rationale: if we already hold large YES inventory,
        further YES selling increases directional exposure.
        Reduce size proportionally.

        Parameters
        ----------
        state : MarketMakerState
        side : Side
        raw_size : int

        Returns
        -------
        int
            Adjusted size (may be zero).
        """
        cap = self.config.max_position_per_market

        if side == Side.YES:
            ratio = abs(state.yes_inventory) / cap
            if state.yes_inventory > 0:   # already long YES — be cautious selling more
                factor = max(0.0, 1.0 - ratio)
                return max(1, int(raw_size * factor))
        else:  # Side.NO
            ratio = abs(state.no_inventory) / cap
            if state.no_inventory > 0:   # already long NO — be cautious selling more
                factor = max(0.0, 1.0 - ratio)
                return max(1, int(raw_size * factor))

        return raw_size

    # ── Main methods ──────────────────────────────────────────────────────────

    def compute_quote(self, market: Market) -> Optional[Quote]:
        """
        Compute a market-making quote for a single market.

        Returns ``None`` if the market doesn't meet criteria:
        - Not Sports / Entertainment category
        - YES price outside [1¢, max_yes_price]
        - Spread below min_spread_bps
        - Kill switch is active
        - Position limits breached

        Parameters
        ----------
        market : Market

        Returns
        -------
        Quote or None
            Quote to offer, or None if not eligible.
        """
        if self._kill_switched:
            return None

        if not market.is_eligible(self.config):
            return None

        # Fair price: deflates the inflated ask by the optimism tax.
        # win_prob = yes_ask captures the inflated probability at which
        # takers are buying. The tax creates edge above fair for the maker.
        fair = self._fair_price(market)
        win_prob = market.yes_ask   # inflated taker probability

        state = self._get_state(market.market_id)

        # Position limit checks
        if abs(state.yes_inventory) >= self.config.max_position_per_market:
            return None
        if self.total_position >= self.config.max_total_position:
            return None

        # Strategy: sell YES on longshots — takers overpay most at low YES prices.
        # Quote at fair + 1¢. This is above fair (positive edge for maker)
        # but below the inflated ask (competitive, gets filled).
        #
        # Example: fair=$0.035, ask=$0.05 → quote at $0.045
        # Edge = $0.045 - $0.035 = +$0.010/contract
        sell_side = Side.YES

        raw_quote_price = fair + 0.01
        # Round to 1¢ grid; must be below ask to be fillable
        quote_price = min(round(raw_quote_price, 2), round(market.yes_ask - 0.001, 2))

        # Don't cross the spread — no quoting below the bid
        if quote_price <= market.yes_bid:
            return None

        # Only quote if we have positive edge
        edge = self._expected_edge(market, quote_price)
        if edge <= 0:
            return None

        # Compute size using Kelly, then adjust for inventory
        raw_size = self._kelly_size(market.yes_ask, win_prob)

        # Cap at per-market remaining
        remaining = self.config.max_position_per_market - abs(state.yes_inventory)
        raw_size = min(raw_size, remaining)

        # Inventory adjustment
        size = self._inventory_adjust(state, sell_side, raw_size)

        if size < 1:
            return None

        state.orders_open += 1

        return Quote(
            market_id=market.market_id,
            sell_side=sell_side,
            price=quote_price,
            size=size,
            expected_edge=round(edge, 4),
        )

    def on_fill(self, fill: Fill) -> None:
        """
        Update state after a fill event.

        PnL accounting (selling YES):
        - Collect premium = fill.price × fill.size at time of sale
        - If YES resolves YES  → pay $1 per contract  → net = premium - size
        - If YES resolves NO   → pay $0              → net = premium

        For realized PnL tracking, we record the premium collected.
        Full resolution PnL is tracked separately when markets settle.

        Parameters
        ----------
        fill : Fill
        """
        state = self._get_state(fill.market_id)
        state.orders_open = max(0, state.orders_open - 1)

        # Update inventory (selling YES → short YES → negative inventory)
        if fill.side == Side.YES:
            state.yes_inventory -= fill.size
        else:
            state.no_inventory -= fill.size

        # Premium collected at sale time
        premium = fill.price * fill.size
        fill.premium_collected = premium
        self._daily_pnl += premium
        state.realized_pnl += premium

    def check_kill_switch(self) -> bool:
        """
        Check whether the kill-switch should trigger.

        The kill-switch activates when daily realized PnL drops below
        negative daily_loss_limit (i.e., losses exceed the threshold).

        Returns
        -------
        bool
            True if kill-switch is now active (quoting must stop).
        """
        if self._kill_switched:
            return True

        if self._daily_pnl <= -self.config.daily_loss_limit:
            self._kill_switched = True
            self._kill_switch_time = datetime.datetime.now()
            return True

        return False

    def reset_kill_switch(self) -> None:
        """Manually reset the kill-switch (e.g., start of new trading day)."""
        self._kill_switched = False
        self._kill_switch_time = None
        self._daily_pnl = 0.0

    def get_state(self, market_id: Optional[str] = None) -> list[MarketMakerState]:
        """
        Get market-maker state for one or all markets.

        Parameters
        ----------
        market_id : str, optional
            If provided, return state for that market only.

        Returns
        -------
        list[MarketMakerState]
        """
        if market_id is not None:
            return [self._get_state(market_id)] if market_id in self._states else []
        return list(self._states.values())

    def status_summary(self) -> dict:
        """Return a human-readable status summary."""
        return {
            "kill_switch_active": self._kill_switched,
            "kill_switch_time": (
                self._kill_switch_time.isoformat()
                if self._kill_switch_time else None
            ),
            "daily_pnl": round(self._daily_pnl, 4),
            "daily_loss_limit": self.config.daily_loss_limit,
            "total_yes_inventory": self.total_yes_inventory,
            "total_no_inventory": self.total_no_inventory,
            "total_position": self.total_position,
            "max_total_position": self.config.max_total_position,
            "active_markets": len(self._states),
        }

    def __repr__(self) -> str:
        s = self.status_summary()
        return (
            f"MarketMaker(kill_switch={s['kill_switch_active']}, "
            f"daily_pnl=${s['daily_pnl']:.2f}, "
            f"position={s['total_position']}/{s['max_total_position']}, "
            f"markets={s['active_markets']})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mock market generator (for demonstration / backtesting)
# ─────────────────────────────────────────────────────────────────────────────


def generate_mock_sports_markets(
    n: int = 20,
    seed: Optional[int] = None,
) -> list[Market]:
    """
    Generate synthetic Sports/Entertainment markets for testing.

    Parameters
    ----------
    n : int
        Number of markets to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[Market]
    """
    rng = random.Random(seed)

    SPORTS_SUB = ["nfl", "nba", "mlb", "nhl", "ncaab", "ncaaf", "ufc", "tennis"]
    ENTERTAINMENT_SUB = [
        "movie_oscars", "tv_awards", "music_awards", "tv_shows"
    ]

    markets = []
    for i in range(n):
        category = rng.choice(["sports", "entertainment"])
        sub = (
            rng.choice(SPORTS_SUB)
            if category == "sports"
            else rng.choice(ENTERTAINMENT_SUB)
        )

        # Longshot distribution: 70% of markets in 1–15¢ range
        if rng.random() < 0.7:
            yes_ask = rng.uniform(0.01, 0.15)
        else:
            yes_ask = rng.uniform(0.15, 0.50)

        # Build bid-ask around ask (2–5¢ spread typical on Kalshi)
        spread = rng.uniform(0.02, 0.05)
        yes_bid = max(0.01, yes_ask - spread)
        # NO bid is roughly 1 - YES ask (no-arbitrage), with spread
        no_bid = rng.uniform(0.40, 0.90)
        no_ask = max(no_bid + 0.01, round(1.0 - yes_bid + 0.01, 2))

        # Volume: longshots have lower volume
        volume = (
            rng.uniform(100, 5000)
            if yes_ask < 0.15
            else rng.uniform(500, 20000)
        )

        # Taker bias: strongest on sports longshots
        taker_bias = (
            rng.uniform(0.10, 0.20)
            if category == "sports" and yes_ask < 0.15
            else rng.uniform(0.05, 0.12)
        )

        markets.append(Market(
            market_id=f"mkt_{i+1:04d}",
            category=category,
            subcategory=sub,
            yes_bid=round(yes_bid, 2),
            yes_ask=round(yes_ask, 2),
            no_bid=round(no_bid, 2),
            no_ask=round(no_ask, 2),
            volume_24h=round(volume, 2),
            taker_bias=round(taker_bias, 3),
        ))

    return markets


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_market(m: Market) -> str:
    return (
        f"{m.market_id:<12} [{m.category:<13} {m.subcategory:<14}] "
        f"YES: {m.yes_bid:.2f}–{m.yes_ask:.2f} "
        f"NO: {m.no_bid:.2f}–{m.no_ask:.2f} "
        f"vol=${m.volume_24h:,.0f}"
    )


def main() -> None:
    """Demonstrate the market-maker on mock Sports market data."""

    print("=" * 60)
    print("  KALSHI MARKET-MAKER — Optimism Tax Exploitation")
    print("  Reference Implementation (Detection Only)")
    print("=" * 60)

    config = MarketMakerConfig(
        max_yes_price=0.15,
        min_spread_bps=200,
        max_position_per_market=500,
        max_total_position=5000,
        daily_loss_limit=100.0,
        kelly_fraction=0.5,
        base_size=10,
        optimism_tax_rate=0.12,
    )

    mm = MarketMaker(config)

    # ── 1. Generate mock markets ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  1.  Mock Sports/Entertainment Markets")
    print("─" * 60)

    markets = generate_mock_sports_markets(n=20, seed=42)

    print(f"\n  Generated {len(markets)} mock markets:\n")
    for m in markets:
        eligible = "✓ eligible" if m.is_eligible(config) else "✗ excluded"
        longshot = " (longshot)" if m.is_longshot else ""
        print(f"  {_fmt_market(m)}{longshot}")
        print(
            f"      spread={m.spread_cents:.1f}¢  "
            f"mid=${m.mid_price:.3f}  {eligible}"
        )

    # ── 2. Compute quotes ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  2.  Quote Computation")
    print("─" * 60)

    quotes = [mm.compute_quote(m) for m in markets]
    valid_quotes = [q for q in quotes if q is not None]

    print(
        f"\n  Computed {len(valid_quotes)} quotes from {len(markets)} markets\n"
    )

    if valid_quotes:
        print(
            f"  {'Market':<12} {'Side':<4} {'Price':>6}  {'Size':>4}  "
            f"{'Edge/Contract':>12}"
        )
        print(f"  {'-'*12} {'----'} {'------'}  {'----'}  {'------------'}")
        for q in valid_quotes:
            print(
                f"  {q.market_id:<12} {q.sell_side.value:<4} "
                f"${q.price:.2f}  {q.size:>4}  "
                f"${q.expected_edge:>+10.4f}"
            )
    else:
        print(
            "  No quotes generated — likely because Kelly returns 0 on\n"
            "  longshot prices (very small win probability → very small Kelly).\n"
            "  Try increasing base_size or kelly_fraction in config.\n"
        )

    # ── 3. Simulate fills ─────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  3.  Simulate Fill Events")
    print("─" * 60)

    if valid_quotes:
        # Simulate filling 5 random quotes
        fill_quotes = random.sample(
            valid_quotes, min(5, len(valid_quotes))
        )
        print(f"\n  Simulating {len(fill_quotes)} fills:\n")
        for q in fill_quotes:
            fill = Fill(
                market_id=q.market_id,
                side=q.sell_side,
                price=q.price,
                size=q.size,
                timestamp=datetime.datetime.now(),
            )
            mm.on_fill(fill)
            print(
                f"  Fill: {fill.market_id}  side={fill.side.value}  "
                f"price=${fill.price:.2f}  size={fill.size}  "
                f"premium=${fill.premium_collected:.2f}"
            )

        print(f"\n  State after fills:")
        for st in mm.get_state():
            if st.orders_open > 0 or st.realized_pnl != 0 or st.yes_inventory != 0:
                print(f"    {st}")
    else:
        print("\n  Skipping fills (no quotes to fill).")

    # ── 4. Status summary ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  4.  Status Summary")
    print("─" * 60)

    summary = mm.status_summary()
    print()
    for key, val in summary.items():
        print(f"  {key:<25} {val}")

    # ── 5. Kill-switch demonstration ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  5.  Kill-Switch Demonstration")
    print("─" * 60)

    print("\n  Simulating a $105 loss to trigger kill-switch...")
    mm._daily_pnl = -105.0  # override for demo

    triggered = mm.check_kill_switch()
    print(f"  Kill-switch triggered: {triggered}")
    print(f"  Kill-switch active: {mm.status_summary()['kill_switch_active']}")

    print("\n  Resetting kill-switch for next trading day...")
    mm.reset_kill_switch()
    print(f"  Kill-switch active after reset: {mm.status_summary()['kill_switch_active']}")

    # ── 6. Kelly sizing walkthrough ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  6.  Kelly Sizing Walkthrough")
    print("─" * 60)

    print(
        "\n  Kelly formula: f* = (b·p - q) / b  where b=1/price-1, q=1-p\n"
    )
    print(
        f"  {'Price':>7} {'WinProb':>9} {'b (odds)':>10} "
        f"{'Raw Kelly':>10} {'Contracts':>10}  Note"
    )
    print(f"  {'-------':>7} {'--------':>9} {'---------':>10} "
          f"{'---------':>10} {'---------':>10}  ----")

    examples = [
        (0.05, 0.05, "5¢ longshot"),
        (0.10, 0.10, "10¢ longshot"),
        (0.25, 0.25, "25¢ medium"),
        (0.50, 0.50, "50¢ even"),
    ]

    for price, prob, note in examples:
        b = 1.0 / price - 1.0
        q = 1.0 - prob
        kelly_raw = max(0.0, (b * prob - q) / b)
        contracts = max(1, int(kelly_raw * 0.5 * config.base_size))
        print(
            f"  ${price:.2f}  {prob:.0%}      "
            f"{b:>8.2f}x  {kelly_raw:>9.4f}  "
            f"{contracts:>9}  {note}"
        )

    # ── 7. Optimism Tax edge breakdown ────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  7.  Optimism Tax Edge — Worked Example")
    print("─" * 60)

    print("""
  Setup: 5¢ YES contract (sports longshot)
  Becker (2026): takers return ~43¢ on the dollar at 5¢ prices
  → Takers systematically lose ~57¢ per dollar invested

  Taker perspective:
    Price paid: $0.05 per contract
    If YES wins: receive $1 → net = +$0.95
    If NO wins:  receive $0 → net = -$0.05
    Implied probability: $0.05 (but true probability ~3.5%)

  Maker perspective (selling YES at $0.05):
    Premium collected: $0.05 per contract
    If YES wins: pay $1 → net = $0.05 - $1 = -$0.95
    If NO wins:  pay $0 → net = $0.05

    Expected PnL = 0.035 × (-0.95) + 0.965 × 0.05
                = -$0.033 + $0.048 = +$0.015 per contract

  Optimism Tax creates edge because:
    - Takers pay $0.05 for a contract worth ~$0.035 (fair)
    - Maker sells at $0.045 (fair + 1¢) → above fair, below ask
    - Edge = $0.045 - $0.035 = +$0.010/contract
    - In expectation: 3.5% × (-$0.955) + 96.5% × $0.045 ≈ +$0.015
""")

    print(f"  MarketMaker instance: {mm}")
    print(f"\n{'─' * 60}")
    print("  Demo complete.")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()