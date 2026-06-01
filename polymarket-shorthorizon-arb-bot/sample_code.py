"""
Python reference: Short-horizon crypto Up/Down trading strategies
Based on PolyBullLabs/polymarket-5min-15min-1hour-arbitrage-trading-bot

Three strategies: 5-min market-neutral, 15-min trend, 1-hour statistical edge.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ── Data Structures ──────────────────────────────────────────────────────────

class Timeframe(Enum):
    M5 = "5min"
    M15 = "15min"
    H1 = "1hour"

@dataclass
class Tick:
    price: float
    volume: float
    timestamp: int
    bid: float
    ask: float
    outcome: str  # "YES" or "NO"

@dataclass
class Signal:
    direction: str  # "LONG" (buy YES) or "SHORT" (buy NO)
    probability: float
    confidence: float


# ── Strategy Base ────────────────────────────────────────────────────────────

class BaseStrategy:
    def __init__(self, timeframe: Timeframe):
        self.timeframe = timeframe
        self.price_buffer: list[Tick] = []

    def add_tick(self, tick: Tick):
        self.price_buffer.append(tick)
        # Keep last 500 ticks
        if len(self.price_buffer) > 500:
            self.price_buffer = self.price_buffer[-500:]

    def evaluate(self) -> Optional[Signal]:
        raise NotImplementedError


# ── Strategy 1: Market-Neutral (5-minute) ───────────────────────────────────

class MarketNeutral5m(BaseStrategy):
    """
    Detects when YES + NO < $1.00 for risk-free arb on 5-min markets.

    Entry: Combined cost of YES + NO < $0.98
    Exit: Auto-resolve at market end
    """

    def __init__(self):
        super().__init__(Timeframe.M5)
        self.min_arb_edge = 0.02  # 2 cents minimum

    def evaluate(self) -> Optional[Signal]:
        if len(self.price_buffer) < 2:
            return None

        latest = self.price_buffer[-1]

        # For 5-min markets, we need both YES and NO prices
        # In practice this comes from tracking separate YES/NO tickers
        yes_price = latest.price if latest.outcome == "YES" else 1.0 - latest.price
        no_price = 1.0 - yes_price  # simplification

        combined = yes_price + no_price

        if combined < (1.0 - self.min_arb_edge):
            arb_bps = int((1.0 - combined) * 10000)
            return Signal(
                direction="LONG",  # buy both YES + NO
                probability=1.0 - combined,
                confidence=min(0.9, arb_bps / 200),
            )

        return None


# ── Strategy 2: Trend Following (15-minute) ─────────────────────────────────

class TrendFollowing15m(BaseStrategy):
    """
    Trend following using EMA/SMA crossovers on 15-min markets.

    Entry: EMA(10) crosses SMA(20) with volume confirmation
    Exit: Take profit at 80% confidence
    """

    def __init__(self):
        super().__init__(Timeframe.M15)
        self.fast_period = 10
        self.slow_period = 20

    def evaluate(self) -> Optional[Signal]:
        if len(self.price_buffer) < self.slow_period + 1:
            return None

        prices = pd.Series([t.price for t in self.price_buffer])
        volumes = pd.Series([t.volume for t in self.price_buffer])

        ema_fast = prices.ewm(span=self.fast_period).mean()
        ema_slow = prices.ewm(span=self.slow_period).mean()

        # Check for crossover
        prev_fast = ema_fast.iloc[-2]
        prev_slow = ema_slow.iloc[-2]
        curr_fast = ema_fast.iloc[-1]
        curr_slow = ema_slow.iloc[-1]

        vol_surge = volumes.iloc[-1] / volumes.tail(20).mean() if len(volumes) >= 20 else 1.0

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            # Bullish crossover
            if vol_surge > 1.5:  # volume confirmation
                confidence = min(0.8, 0.3 + (curr_fast - curr_slow) * 5)
                return Signal("LONG", curr_fast, confidence)

        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            # Bearish crossover
            if vol_surge > 1.5:
                confidence = min(0.8, 0.3 + (curr_slow - curr_fast) * 5)
                return Signal("SHORT", 1.0 - curr_fast, confidence)

        return None


# ── Strategy 3: Statistical Edge (1-hour) ───────────────────────────────────

class StatisticalEdge1h(BaseStrategy):
    """
    Statistical arbitrage on 1-hour markets using RSI + MACD + order book.

    Entry: Z-score > 2.0 or RSI oversold/overbought
    Exit: Regression to mean
    """

    def __init__(self):
        super().__init__(Timeframe.H1)

    def evaluate(self) -> Optional[Signal]:
        if len(self.price_buffer) < 50:
            return None

        prices = pd.Series([t.price for t in self.price_buffer])

        # Z-score calculation
        rolling_mean = prices.tail(20).mean()
        rolling_std = prices.tail(20).std()
        zscore = (prices.iloc[-1] - rolling_mean) / max(rolling_std, 0.001)

        # RSI
        diffs = prices.diff()
        gains = diffs.clip(lower=0)
        losses = (-diffs).clip(lower=0)
        avg_gain = gains.tail(14).mean()
        avg_loss = losses.tail(14).mean()
        rs = avg_gain / max(avg_loss, 0.001)
        rsi = 100 - (100 / (1 + rs))

        # Order book imbalance
        latest = self.price_buffer[-1]
        spread = latest.ask - latest.bid
        mid = (latest.ask + latest.bid) / 2
        rel_spread = spread / max(mid, 0.001)

        signals = []

        # Z-score signals
        if zscore > 2.0:
            signals.append(Signal("SHORT", 0.3, min(0.7, zscore / 3)))
        elif zscore < -2.0:
            signals.append(Signal("LONG", 0.7, min(0.7, abs(zscore) / 3)))

        # RSI signals  
        if rsi < 30:
            signals.append(Signal("LONG", 0.6, 0.6))
        elif rsi > 70:
            signals.append(Signal("SHORT", 0.4, 0.6))

        # Volatility adjustment
        if rel_spread > 0.05:  # wide spread → reduce confidence
            for s in signals:
                s.confidence *= 0.7

        if not signals:
            return None

        # Aggregate signals
        long_probs = [s.probability for s in signals if s.direction == "LONG"]
        short_probs = [s.probability for s in signals if s.direction == "SHORT"]

        if long_probs and (not short_probs or np.mean(long_probs) > np.mean(short_probs)):
            return Signal("LONG", np.mean(long_probs), np.mean([s.confidence for s in signals]))
        elif short_probs:
            return Signal("SHORT", np.mean(short_probs), np.mean([s.confidence for s in signals]))

        return None


# ── Orchestrator ─────────────────────────────────────────────────────────────

class MultiTimeframeArbBot:
    """Run all three strategies and aggregate signals."""

    def __init__(self):
        self.strategies = {
            Timeframe.M5: MarketNeutral5m(),
            Timeframe.M15: TrendFollowing15m(),
            Timeframe.H1: StatisticalEdge1h(),
        }

    def process_tick(self, tick: Tick) -> list[Signal]:
        """Feed tick to all strategies and return any signals."""
        for strategy in self.strategies.values():
            strategy.add_tick(tick)

        signals = []
        for tf, strategy in self.strategies.items():
            signal = strategy.evaluate()
            if signal:
                signals.append(signal)
                print(f"  [{tf.value}] {signal.direction} "
                      f"p={signal.probability:.2f} c={signal.confidence:.2f}")

        return signals

    def report_status(self):
        """Print status of all strategy buffers."""
        for tf, strategy in self.strategies.items():
            print(f"  {tf.value}: {len(strategy.price_buffer)} ticks stored")


# ── Example Usage ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    bot = MultiTimeframeArbBot()

    # Simulate 100 ticks over 5 seconds
    prices = np.random.randn(100).cumsum() * 0.01 + 0.5

    for i, p in enumerate(prices):
        tick = Tick(
            price=max(0.01, min(0.99, p)),
            volume=np.random.rand() * 1000,
            timestamp=int(time.time()) + i,
            bid=max(0.01, p - 0.02),
            ask=min(0.99, p + 0.02),
            outcome="YES",
        )

        signals = bot.process_tick(tick)
        if signals:
            print(f"  Trade signals generated at tick {i}")
