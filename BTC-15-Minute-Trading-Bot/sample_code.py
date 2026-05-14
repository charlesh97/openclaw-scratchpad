"""
Reference: https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot

Simplified illustration of the 7-phase architecture's risk management component.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class RiskConfig:
    max_drawdown_pct: float = 0.10       # 10% max drawdown
    kelly_fraction: float = 0.25          # Fractional Kelly for sizing
    max_position_pct: float = 0.05        # 5% of capital per position
    stop_loss_pct: float = 0.02           # 2% stop loss per trade
    min_win_probability: float = 0.55     # Minimum estimated win prob


class RiskManager:
    """
    Professional risk management module from the 7-phase architecture.
    """

    def __init__(self, capital: float, config: RiskConfig):
        self.initial_capital = capital
        self.current_capital = capital
        self.config = config
        self.peak_capital = capital

    def compute_position_size(
        self,
        estimated_win_prob: float,
        potential_return: float,
        potential_loss: float,
    ) -> Optional[float]:
        """Compute Kelly-optimal position size with constraints."""

        if estimated_win_prob < self.config.min_win_probability:
            return 0.0

        # Kelly formula: f* = (p * b - q) / b
        # where p = win prob, b = win/loss ratio, q = 1-p
        b = potential_return / potential_loss if potential_loss > 0 else 0
        p = estimated_win_prob
        q = 1 - p

        kelly_pct = (p * b - q) / b if b > 0 else 0
        kelly_pct = max(0, kelly_pct)

        # Apply fractional Kelly and position cap
        position_pct = min(
            kelly_pct * self.config.kelly_fraction,
            self.config.max_position_pct,
        )

        position_value = self.current_capital * position_pct

        # Check drawdown constraint
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > self.config.max_drawdown_pct:
            return 0.0  # Trading halted at max drawdown

        return position_value

    def update_capital(self, pnl: float):
        """Update capital after trade settlement."""
        self.current_capital += pnl
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

    def is_trading_halted(self) -> bool:
        """Check if trading should be halted due to drawdown."""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        return drawdown > self.config.max_drawdown_pct
