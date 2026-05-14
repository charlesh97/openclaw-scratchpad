"""
Reference: https://github.com/gamma-trade-lab/polymarket-copy-trading-bot

Python illustration of the wallet tracking and copy trading decision logic.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timedelta
import statistics


@dataclass
class TraderActivity:
    wallet_address: str
    trades: list = field(default_factory=list)
    win_rate: Optional[float] = None
    avg_position_size: Optional[float] = None
    total_pnl: float = 0.0
    last_active: Optional[datetime] = None


class CopyTraderEngine:
    """
    Tracks top wallets and decides when to mirror their trades.
    """

    def __init__(self, min_win_rate: float = 0.55, min_trades: int = 30):
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades
        self.tracked_wallets: dict = {}

    def evaluate_wallet(self, wallet: str, trade_history: list) -> Optional[float]:
        """
        Score a wallet (0-100) based on historical performance.
        Returns score or None if insufficient data.
        """
        if len(trade_history) < self.min_trades:
            return None

        resolved = [t for t in trade_history if t.get("resolved")]
        if not resolved:
            return None

        wins = sum(1 for t in resolved if t.get("pnl", 0) > 0)
        win_rate = wins / len(resolved)

        if win_rate < self.min_win_rate:
            return None

        avg_win = statistics.mean(
            [t["pnl"] for t in resolved if t["pnl"] > 0]
        ) if any(t["pnl"] > 0 for t in resolved) else 0
        avg_loss = abs(statistics.mean(
            [t["pnl"] for t in resolved if t["pnl"] < 0]
        )) if any(t["pnl"] < 0 for t in resolved) else float("inf")

        profit_factor = avg_win / avg_loss if avg_loss > 0 else 2.0

        # Score: win rate 0-50pts + profit factor 0-50pts
        score = (win_rate * 50) + min(profit_factor * 25, 50)
        return round(score, 1)

    def should_copy_trade(self, wallet: str, trade: dict, score: float) -> bool:
        """
        Decision: should we mirror this trade from this wallet?
        """
        threshold = 60  # minimum score threshold
        return score >= threshold
