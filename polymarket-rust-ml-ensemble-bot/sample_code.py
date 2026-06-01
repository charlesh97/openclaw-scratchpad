"""
Python reference: ML Ensemble Signal Pipeline for Polymarket
Based on skharchikov/polymarket-bot architecture

This implements the core signal pipeline in Python for reference:
  1. Feature engineering (29 features)
  2. XGBoost ensemble prediction
  3. LLM consensus estimate
  4. Bayesian updating
  5. Kelly position sizing
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class GammaMarket:
    condition_id: str
    token_id: str
    outcome: str  # "YES" or "NO"
    price: float
    volume_24h: float
    liquidity: float
    open_interest: float

@dataclass
class PriceTick:
    timestamp: int
    price: float
    volume: float
    bid: float
    ask: float

@dataclass
class Signal:
    source: str          # "XgBoost", "LlmConsensus", "Bayesian"
    probability: float   # 0.0 to 1.0
    confidence: float    # 0.0 to 1.0
    timestamp: int

# ── Feature Engineering (v5 — 29 features) ────────────────────────────────

class MarketFeatures:
    """Engineer 29 features from market data for XGBoost ensemble."""

    @staticmethod
    def compute(prices: list[PriceTick], market: GammaMarket) -> dict:
        if len(prices) < 2:
            return {}

        df = pd.DataFrame([{
            'price': p.price,
            'volume': p.volume,
            'bid': p.bid,
            'ask': p.ask,
            'spread': p.ask - p.bid,
        } for p in prices])

        features = {}

        # 1. Price features
        features['price_latest'] = df['price'].iloc[-1]
        features['price_mean_10'] = df['price'].tail(10).mean()
        features['price_mean_50'] = df['price'].tail(50).mean() if len(df) >= 50 else df['price'].mean()
        features['price_std_10'] = df['price'].tail(10).std()
        features['price_std_50'] = df['price'].tail(50).std() if len(df) >= 50 else df['price'].std()
        features['price_momentum_5'] = df['price'].diff(5).iloc[-1] if len(df) > 5 else 0
        features['price_momentum_20'] = df['price'].diff(20).iloc[-1] if len(df) > 20 else 0
        features['price_zscore'] = (df['price'].iloc[-1] - df['price'].mean()) / (df['price'].std() + 1e-8)

        # 2. Volume features
        features['volume_latest'] = df['volume'].iloc[-1]
        features['volume_mean_10'] = df['volume'].tail(10).mean()
        features['volume_ratio'] = df['volume'].iloc[-1] / (df['volume'].mean() + 1e-8)

        # 3. Order book features
        features['spread'] = df['spread'].iloc[-1]
        features['spread_mean_10'] = df['spread'].tail(10).mean()
        features['mid_price'] = (df['bid'].iloc[-1] + df['ask'].iloc[-1]) / 2
        features['spread_bps'] = features['spread'] / (features['mid_price'] + 1e-8) * 10000

        # 4. Market metadata features
        features['volume_24h'] = market.volume_24h
        features['liquidity'] = market.liquidity
        features['open_interest'] = market.open_interest
        features['current_price'] = market.price
        features['price_deviation'] = abs(market.price - features['price_latest']) / (features['price_latest'] + 1e-8)

        # 5. Time-series features
        features['price_min_10'] = df['price'].tail(10).min()
        features['price_max_10'] = df['price'].tail(10).max()
        features['price_range_10'] = features['price_max_10'] - features['price_min_10']

        # 6. Cross-sectional features
        if 'spread' in df.columns and len(df) > 1:
            roll_corr = df['price'].rolling(10).corr(df['spread'])
            features['price_spread_corr'] = roll_corr.iloc[-1] if not np.isnan(roll_corr.iloc[-1]) else 0

        return features

# ── Bayesian Signal Updater ────────────────────────────────────────────────

class BayesianUpdater:
    """Combine ML and LLM signals using Bayesian updating."""

    def __init__(self, prior: float = 0.5):
        self.prior = prior

    def update(self, ml_signal: Signal, llm_signal: Signal,
               ml_likelihood_ratio: float = 2.0,
               llm_likelihood_ratio: float = 1.5) -> Signal:
        """
        Combine ML and LLM probability estimates using Bayes' theorem.

        P(event | ML, LLM) ∝ P(event) * LR(ML) * LR(LLM)

        where LR = P(signal | event) / P(signal | ¬event)
        """
        prior = self.prior

        # Update with ML signal
        if ml_signal and ml_signal.confidence > 0.3:
            odds = prior / (1 - prior + 1e-8)
            lr = ml_likelihood_ratio if ml_signal.probability > 0.5 else 1.0 / ml_likelihood_ratio
            odds *= lr
            post_ml = odds / (1 + odds)
        else:
            post_ml = prior

        # Update with LLM signal
        if llm_signal and llm_signal.confidence > 0.3:
            odds = post_ml / (1 - post_ml + 1e-8)
            lr = llm_likelihood_ratio if llm_signal.probability > 0.5 else 1.0 / llm_likelihood_ratio
            odds *= lr
            posterior = odds / (1 + odds)
        else:
            posterior = post_ml

        combined_confidence = min(1.0, (ml_signal.confidence + llm_signal.confidence) / 2)

        return Signal(
            source="Bayesian",
            probability=posterior,
            confidence=combined_confidence,
            timestamp=ml_signal.timestamp,
        )

# ── Kelly Criterion Position Sizer ─────────────────────────────────────────

class KellySizer:
    """Position sizing using the Kelly Criterion."""

    def __init__(self, bankroll: float, risk_profile: str = "balanced"):
        self.bankroll = bankroll
        self.risk_profiles = {
            "conservative": 0.25,  # quarter Kelly
            "balanced": 0.5,       # half Kelly
            "aggressive": 1.0,     # full Kelly
        }
        self.kelly_fraction = self.risk_profiles.get(risk_profile, 0.5)

    def compute_kelly(self, probability: float, payout_odds: float) -> float:
        """
        Kelly formula: f* = (bp - q) / b
        where b = net odds, p = win prob, q = lose prob
        """
        b = payout_odds  # e.g., 1.0 for even-money (binary shares)
        p = probability
        q = 1 - p

        kelly = (b * p - q) / (b + 1e-8)
        return max(0, min(kelly, 0.25))  # cap at 25% of bankroll

    def size_position(self, signal: Signal, market_price: float) -> float:
        """Calculate position size in USDC."""
        payout_odds = (1 - market_price) / market_price if market_price > 0 else 1.0
        kelly_pct = self.compute_kelly(signal.probability, payout_odds)
        return self.bankroll * kelly_pct * self.kelly_fraction

# ── Full Pipeline Orchestration ────────────────────────────────────────────

class MLEnsemblePipeline:
    """Complete ML ensemble signal pipeline."""

    def __init__(self, bankroll: float = 1000.0, risk_profile: str = "balanced"):
        self.features = MarketFeatures()
        self.bayesian = BayesianUpdater()
        self.sizer = KellySizer(bankroll, risk_profile)
        self.price_buffer: dict[str, list[PriceTick]] = {}

    def process_market(self, market: GammaMarket, prices: list[PriceTick]) -> dict:
        """
        Full pipeline: features → XGBoost → LLM → Bayesian → Kelly sizing.

        Returns trade decision dict.
        """
        # 1. Feature engineering
        feature_vector = self.features.compute(prices, market)
        if not feature_vector:
            return {"action": "skip", "reason": "insufficient data"}

        # 2. XGBoost prediction (would call trained model server)
        # In production: POST /predict to Python FastAPI sidecar
        # Here: placeholder using price-based heuristic
        xgb_prob = self._xgb_predict(feature_vector)
        xgb_signal = Signal("XgBoost", xgb_prob, min(0.8, abs(xgb_prob - 0.5) * 2 + 0.3), prices[-1].timestamp)

        # 3. LLM consensus (would call OpenAI API)
        # In production: fetch news → embed → LLM prompt → probability
        llm_prob = self._llm_estimate(market)
        llm_signal = Signal("LlmConsensus", llm_prob, 0.6, prices[-1].timestamp)

        # 4. Bayesian combine
        combined = self.bayesian.update(xgb_signal, llm_signal)

        # 5. Kelly position sizing
        position_size = self.sizer.size_position(combined, market.price)

        return {
            "action": "buy" if combined.probability > market.price + 0.02 else "hold",
            "market": market.condition_id,
            "outcome": market.outcome,
            "signal": combined,
            "position_size_usdc": position_size,
            "features": feature_vector,
            "xgb_prob": xgb_prob,
            "llm_prob": llm_prob,
            "combined_prob": combined.probability,
            "edge": combined.probability - market.price,
        }

    def _xgb_predict(self, features: dict) -> float:
        """Placeholder: In production, calls FastAPI model server."""
        # Real implementation would POST features to XGBoost model server
        # Returns probability estimate 0.0-1.0
        price = features.get('price_latest', 0.5)
        momentum = features.get('price_momentum_5', 0)
        zscore = features.get('price_zscore', 0)

        # Simple placeholder heuristic
        raw = price + 0.1 * np.tanh(momenton * 5) + 0.05 * np.tanh(zscore)
        return np.clip(raw, 0.01, 0.99)

    def _llm_estimate(self, market: GammaMarket) -> float:
        """Placeholder: In production, calls OpenAI for news-based probability."""
        return 0.5  # neutral estimate


# ── Example Usage ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    market = GammaMarket(
        condition_id="0xabc123",
        token_id="0xdef456",
        outcome="YES",
        price=0.55,
        volume_24h=100000,
        liquidity=50000,
        open_interest=30000,
    )

    prices = [
        PriceTick(int(time.time()) - i * 60, 0.52 + np.random.rand() * 0.06,
                  np.random.rand() * 1000, 0.51, 0.53)
        for i in range(50, 0, -1)
    ]

    pipeline = MLEnsemblePipeline(bankroll=5000, risk_profile="balanced")
    decision = pipeline.process_market(market, prices)

    print("Trade Decision:")
    for k, v in decision.items():
        print(f"  {k}: {v}")
