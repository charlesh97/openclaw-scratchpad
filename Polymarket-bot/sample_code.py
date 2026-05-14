"""
Reference: https://github.com/MrFadiAi/Polymarket-bot

Simplified illustration of the spread arbitrage strategy logic.
This is a structural reconstruction, not a copy of the original.
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class ArbitrageOpportunity:
    market_id: str
    yes_price: float
    no_price: float
    spread: float           # 1.0 - (yes + no) → risk-free profit margin
    max_size: int           # constrained by order book depth
    estimated_profit_usdc: float


class SpreadArbitrageStrategy:
    """
    Detects YES + NO < 1.00 bundle arbitrage opportunities
    and computes executable size from order book liquidity.
    """

    def __init__(self, min_spread: float = 0.005, max_capital_per_trade: float = 500.0):
        self.min_spread = min_spread          # minimum 0.5% risk-free margin
        self.max_capital_per_trade = max_capital_per_trade

    def scan_market(self, order_book: dict) -> Optional[ArbitrageOpportunity]:
        """
        Scan a single binary market for bundle arbitrage.

        order_book: {
            'yes_bids': [(price, size), ...],   # sorted desc
            'no_bids':  [(price, size), ...],   # sorted desc
        }
        """
        if not order_book.get('yes_bids') or not order_book.get('no_bids'):
            return None

        # Best bid for YES = what you'd sell YES at
        best_yes_bid = order_book['yes_bids'][0][0]
        # Best bid for NO = what you'd sell NO at
        best_no_bid = order_book['no_bids'][0][0]

        total = best_yes_bid + best_no_bid
        spread = 1.0 - total

        if spread < self.min_spread:
            return None

        # Compute executable size from minimum depth on either side
        yes_depth = sum(s for _, s in order_book['yes_bids'] if _ >= best_yes_bid)
        no_depth = sum(s for _, s in order_book['no_bids'] if _ >= best_no_bid)
        max_shares = min(yes_depth, no_depth)

        capital_needed = max_shares * total
        if capital_needed > self.max_capital_per_trade:
            max_shares = int(self.max_capital_per_trade / total)
            capital_needed = max_shares * total

        if max_shares < 1:
            return None

        return ArbitrageOpportunity(
            market_id="",
            yes_price=best_yes_bid,
            no_price=best_no_bid,
            spread=spread,
            max_size=max_shares,
            estimated_profit_usdc=max_shares * spread,
        )

    def execute_bundle(self, opportunity: ArbitrageOpportunity, clob_api) -> bool:
        """
        Execute the bundle: sell YES and sell NO simultaneously.
        Two independent limit orders; risk is fully hedged.
        """
        if opportunity.estimated_profit_usdc <= 0:
            return False

        # Place both orders atomically (in practice: via CLOB batch endpoint)
        yes_order = clob_api.place_order(
            market=opportunity.market_id,
            side="SELL",
            price=opportunity.yes_price,
            size=opportunity.max_size,
        )
        no_order = clob_api.place_order(
            market=opportunity.market_id,
            side="SELL",
            price=opportunity.no_price,
            size=opportunity.max_size,
        )

        return yes_order.get("success") and no_order.get("success")


# Strategy performance monitor for auto-rotation
class StrategyMonitor:
    """
    Tracks rolling PnL per strategy to feed the auto-rotation system.
    """

    def __init__(self, window_hours: int = 24):
        self.trades: list = []
        self.window = window_hours

    def record_trade(self, strategy_name: str, pnl: float, timestamp: float = None):
        self.trades.append({
            "strategy": strategy_name,
            "pnl": pnl,
            "ts": timestamp or time.time(),
        })

    def rolling_sharpe(self, strategy_name: str) -> float:
        """Simple rolling Sharpe approximation for a strategy."""
        relevant = [
            t["pnl"] for t in self.trades
            if t["strategy"] == strategy_name
            and time.time() - t["ts"] < self.window * 3600
        ]
        if len(relevant) < 10:
            return 0.0
        mean_pnl = sum(relevant) / len(relevant)
        std_pnl = (sum((x - mean_pnl) ** 2 for x in relevant) / len(relevant)) ** 0.5
        return mean_pnl / std_pnl if std_pnl > 0 else 0.0
