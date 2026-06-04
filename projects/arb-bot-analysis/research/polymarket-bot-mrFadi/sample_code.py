"""
Reference implementation sketch: Polymarket Bot - DipArb + Copy Trading.
Conceptual Python translation of the Node.js bot's core logic.
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Trader:
    address: str
    win_rate: float
    profit_factor: float
    total_trades: int
    consistency_score: float


@dataclass
class TradeSignal:
    strategy: str  # "dip_arb", "copy_trade", "smart_money", "market_making"
    market_id: str
    side: str  # "YES" or "NO"
    entry_price: float
    size: float
    confidence: float


class SmartMoneyFilter:
    """Only follow traders meeting quality thresholds."""
    MIN_WIN_RATE = 0.60
    MIN_PROFIT_FACTOR = 1.5
    MIN_CONSISTENCY = 0.7
    
    def filter_traders(self, traders: list[Trader]) -> list[Trader]:
        return [
            t for t in traders
            if t.win_rate >= self.MIN_WIN_RATE
            and t.profit_factor >= self.MIN_PROFIT_FACTOR
            and t.consistency_score >= self.MIN_CONSISTENCY
        ]


class PositionSizer:
    """Dynamic sizing: reduce during losses, increase during wins."""
    
    def compute_size(self, base_size: float, 
                     consecutive_losses: int,
                     consecutive_wins: int,
                     portfolio_pnl: float) -> float:
        if portfolio_pnl < -0.05 * base_size * 20:  # -5% of base
            return 0  # Halt
        loss_penalty = 0.5 ** consecutive_losses
        win_bonus = 1.2 ** min(consecutive_wins, 5)
        return base_size * loss_penalty * win_bonus


class RiskManager:
    DAILY_LOSS_LIMIT = 0.05  # 5%
    MONTHLY_LOSS_LIMIT = 0.15  # 15%
    DRAWDOWN_LIMIT = 0.25  # 25%
    TOTAL_HALT = 0.40  # 40%
    
    def check_limits(self, portfolio: dict) -> bool:
        """Return True if trading should continue."""
        dd = portfolio.get("drawdown", 0)
        daily = portfolio.get("daily_pnl_pct", 0)
        monthly = portfolio.get("monthly_pnl_pct", 0)
        total = portfolio.get("total_pnl_pct", 0)
        
        if dd < -self.DRAWDOWN_LIMIT:
            return False
        if total < -self.TOTAL_HALT:
            return False
        if daily < -self.DAILY_LOSS_LIMIT:
            return False
        if monthly < -self.MONTHLY_LOSS_LIMIT:
            return False
        return True


class DipArbStrategy:
    """Buy when price drops abnormally on short timeframes."""
    
    def detect_dip(self, prices: list[float], threshold: float = 0.15) -> Optional[TradeSignal]:
        if len(prices) < 10:
            return None
        recent_ma = sum(prices[-5:]) / 5
        current = prices[-1]
        drop = (recent_ma - current) / recent_ma
        if drop > threshold:
            return TradeSignal(
                strategy="dip_arb",
                market_id="",
                side="YES" if drop > 0 else "NO",
                entry_price=current,
                size=1.0,
                confidence=min(1.0, drop * 3),
            )
        return None
