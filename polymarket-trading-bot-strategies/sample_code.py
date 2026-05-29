"""
PolyHFT — 10-Strategy Polymarket Trading Bot (Reference Architecture)
Based on: https://github.com/Anmoldureha/polymarket-trading-bot-strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MarketData:
    market_id: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume_24h: float
    liquidity: float


class TradingStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, max_position: float = 15.0):
        self.name = name
        self.max_position = max_position
        self.current_position = 0.0
    
    @abstractmethod
    def evaluate(self, market: MarketData) -> Optional[dict]:
        """Evaluate market and return trade signal or None."""
        pass
    
    @abstractmethod
    def risk_check(self, signal: dict) -> bool:
        """Run strategy-specific risk checks."""
        pass


class MicroSpreadStrategy(TradingStrategy):
    """Capture bid-ask spread with tight orders."""
    
    def evaluate(self, market: MarketData) -> Optional[dict]:
        spread = market.yes_ask - market.yes_bid
        if spread > 0.02:  # 2 cent spread threshold
            return {
                'action': 'limit_bid',
                'price': market.yes_bid + 0.005,
                'size': min(5.0, self.max_position - abs(self.current_position))
            }
        return None
    
    def risk_check(self, signal: dict) -> bool:
        return abs(self.current_position + signal['size']) <= self.max_position


class CombinatorialArbitrageStrategy(TradingStrategy):
    """
    Detect arbitrage across multiple related markets.
    e.g., Team A wins Game 1 + Team A wins Game 2 ≠ Team A wins both.
    """
    
    def __init__(self, name: str, max_position: float = 10.0):
        super().__init__(name, max_position)
        self.related_markets = {}  # market_id -> list of related market_ids
    
    def register_related_markets(self, market_id: str, related: List[str]):
        self.related_markets[market_id] = related
    
    def evaluate(self, market: MarketData) -> Optional[dict]:
        related = self.related_markets.get(market.market_id, [])
        if not related:
            return None
        # Check if sum of related probabilities violates bounds
        # (simplified — real implementation would use full probability tree)
        return None  # Placeholder for combinatorial arb logic


class StrategyManager:
    """Runs multiple strategies with independent risk controls."""
    
    def __init__(self):
        self.strategies: List[TradingStrategy] = []
    
    def add_strategy(self, strategy: TradingStrategy):
        self.strategies.append(strategy)
    
    def run_all(self, market: MarketData) -> List[dict]:
        signals = []
        for strategy in self.strategies:
            signal = strategy.evaluate(market)
            if signal and strategy.risk_check(signal):
                signal['strategy'] = strategy.name
                signals.append(signal)
        return signals
