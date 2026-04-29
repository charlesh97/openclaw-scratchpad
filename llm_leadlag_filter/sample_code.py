"""
LLM Semantic Lead-Lag Filter for Prediction Markets

Uses Granger causality to detect statistical lead-lag relationships,
then validates them with an LLM semantic filter to reduce false positives.

Source: arXiv:2602.07048 — "LLM as a Risk Manager" (Feb 2026)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json
import time


@dataclass
class MarketSnapshot:
    """A point-in-time snapshot of a prediction market."""
    market_id: str
    question: str
    yes_price: float  # 0.0 to 1.0
    no_price: float   # 0.0 to 1.0
    timestamp: float
    volume_24h: float = 0.0
    category: str = ""


@dataclass
class LeadLagCandidate:
    """A candidate lead-lag relationship between two markets."""
    lead_market: str
    lag_market: str
    lag_seconds: float
    granger_p_value: float
    cross_correlation: float
    direction: str  # "positive" or "negative"


@dataclass
class ValidatedPair:
    """A lead-lag pair validated by LLM semantic filtering."""
    lead_market: str
    lag_market: str
    lag_seconds: float
    llm_confidence: float  # 0.0 to 1.0
    reasoning: str
    signal_strength: float = 0.0


class GrangerLeadLagDetector:
    """
    Detects lead-lag relationships using time-shifted cross-correlation
    and Granger causality testing.
    """

    def __init__(self, max_lag_seconds: float = 300.0, significance_level: float = 0.05):
        self.max_lag_seconds = max_lag_seconds
        self.significance_level = significance_level

    def detect_lead_lag(
        self,
        lead_prices: list[float],
        lag_prices: list[float],
        timestamps: list[float],
        lead_id: str,
        lag_id: str,
    ) -> list[LeadLagCandidate]:
        """
        Detect lead-lag relationships between two price series.

        Returns candidate pairs where lead market's price changes
        precede lag market's price changes.
        """
        if len(lead_prices) < 10 or len(lag_prices) < 10:
            return []

        lead_arr = np.array(lead_prices)
        lag_arr = np.array(lag_prices)

        # Normalize to returns
        lead_returns = np.diff(lead_arr) / (lead_arr[:-1] + 1e-10)
        lag_returns = np.diff(lag_arr) / (lag_arr[:-1] + 1e-10)

        candidates = []

        # Test different lag offsets
        for lag_offset in range(1, min(30, len(lead_returns) // 3)):
            if lag_offset >= len(lead_returns) or lag_offset >= len(lag_returns):
                break

            # Cross-correlation at this lag
            lead_shifted = lead_returns[:-lag_offset]
            lag_shifted = lag_returns[lag_offset:]

            if len(lead_shifted) < 5:
                continue

            corr = np.corrcoef(lead_shifted, lag_shifted)[0, 1]

            if abs(corr) > 0.3:  # Minimum correlation threshold
                # Estimate lag in seconds from timestamp spacing
                if len(timestamps) > 1:
                    avg_dt = np.median(np.diff(timestamps))
                    lag_seconds = lag_offset * avg_dt
                else:
                    lag_seconds = lag_offset * 60  # Default 1 min per step

                if lag_seconds <= self.max_lag_seconds:
                    candidates.append(LeadLagCandidate(
                        lead_market=lead_id,
                        lag_market=lag_id,
                        lag_seconds=lag_seconds,
                        granger_p_value=0.01,  # Placeholder — real impl uses statsmodels
                        cross_correlation=corr,
                        direction="positive" if corr > 0 else "negative",
                    ))

        return candidates


class LLMSemanticFilter:
    """
    Uses an LLM to validate whether a lead-lag relationship
    makes semantic sense — filtering spurious correlations.

    In production, this calls an LLM API. Here we simulate the logic.
    """

    def __init__(self):
        self.validation_prompt_template = """
You are a prediction market analyst. Evaluate whether this lead-lag relationship
between two prediction markets is logically plausible.

LEAD MARKET: "{lead_question}"
LAG MARKET: "{lag_question}"
Observed lag: {lag_seconds:.0f} seconds
Statistical correlation: {correlation:.3f}

Questions to consider:
1. Does the lead market's outcome logically influence or precede the lag market?
2. Is the observed lag timing plausible for this type of relationship?
3. Could this be a spurious correlation driven by shared keywords?
4. Would this relationship hold under different market conditions?

Respond with JSON:
{{
  "valid": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}
"""

    def validate(self, candidate: LeadLagCandidate,
                 lead_question: str, lag_question: str) -> Optional[ValidatedPair]:
        """
        Validate a lead-lag candidate using LLM semantic analysis.

        In production, this would call an LLM API.
        This implementation returns a simulated validation.
        """
        # Build the prompt
        prompt = self.validation_prompt_template.format(
            lead_question=lead_question,
            lag_question=lag_question,
            lag_seconds=candidate.lag_seconds,
            correlation=candidate.cross_correlation,
        )

        # --- SIMULATED LLM RESPONSE ---
        # In production: call OpenAI/Anthropic/local LLM here
        # response = llm_client.complete(prompt)

        # Simulated validation logic based on heuristics
        confidence = self._heuristic_validation(candidate, lead_question, lag_question)

        if confidence < 0.4:
            return None  # Filtered out

        return ValidatedPair(
            lead_market=candidate.lead_market,
            lag_market=candidate.lag_market,
            lag_seconds=candidate.lag_seconds,
            llm_confidence=confidence,
            reasoning=f"Semantic validation: {'plausible' if confidence > 0.6 else 'weak'} relationship "
                      f"with {candidate.cross_correlation:.2f} correlation over {candidate.lag_seconds:.0f}s lag",
            signal_strength=candidate.cross_correlation * confidence,
        )

    def _heuristic_validation(
        self, candidate: LeadLagCandidate,
        lead_question: str, lag_question: str,
    ) -> float:
        """
        Heuristic validation (simulates what LLM would do).
        Replace with actual LLM call in production.
        """
        score = 0.5  # Base score

        # Higher correlation → higher confidence
        score += min(0.3, abs(candidate.cross_correlation) * 0.5)

        # Shorter lag → more plausible
        if candidate.lag_seconds < 60:
            score += 0.1
        elif candidate.lag_seconds > 180:
            score -= 0.1

        # Penalty for very weak correlations
        if abs(candidate.cross_correlation) < 0.35:
            score -= 0.15

        return max(0.0, min(1.0, score))


class LeadLagSignalGenerator:
    """
    Generates trading signals from validated lead-lag pairs.
    Uses quarter-Kelly position sizing for risk control.
    """

    def __init__(self, kelly_fraction: float = 0.25, max_position: float = 0.10):
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position

    def generate_signal(
        self,
        validated_pair: ValidatedPair,
        lead_price_change: float,
        lead_current_price: float,
    ) -> dict:
        """
        Generate a trading signal when the lead market moves.

        Args:
            validated_pair: The validated lead-lag relationship
            lead_price_change: Recent price change in the lead market
            lead_current_price: Current price of the lead market

        Returns:
            Signal dict with action, position size, confidence
        """
        if abs(lead_price_change) < 0.01:  # Minimum move threshold
            return {"action": "hold", "reason": "Lead move too small"}

        # Direction: if lead went up, predict lag will follow
        direction = "buy_yes" if lead_price_change > 0 else "buy_no"

        # Signal strength = correlation × LLM confidence × lead move magnitude
        raw_signal = (
            validated_pair.signal_strength
            * min(1.0, abs(lead_price_change) / 0.05)  # Normalize move size
        )

        # Quarter-Kelly position sizing
        # Kelly: f* = (p * b - q) / b where p=win prob, b=odds
        # Simplified: use signal_strength as edge estimate
        kelly_size = self.kelly_fraction * raw_signal
        position_size = min(kelly_size, self.max_position)

        return {
            "action": direction,
            "position_size": round(position_size, 4),
            "confidence": round(validated_pair.llm_confidence, 3),
            "signal_strength": round(raw_signal, 4),
            "lead_market": validated_pair.lead_market,
            "lag_market": validated_pair.lag_market,
            "expected_lag_seconds": validated_pair.lag_seconds,
            "reason": f"Lead moved {lead_price_change:+.3f}, "
                      f"lag predicted to follow in ~{validated_pair.lag_seconds:.0f}s",
        }


class LeadLagPipeline:
    """
    Complete pipeline: detect → validate → signal.

    Usage:
        pipeline = LeadLagPipeline()
        # Feed in market snapshots over time
        pipeline.update(market_snapshot)
        # Get signals
        signals = pipeline.get_signals()
    """

    def __init__(self):
        self.detector = GrangerLeadLagDetector()
        self.filter = LLMSemanticFilter()
        self.signal_gen = LeadLagSignalGenerator()
        self.price_history: dict[str, list[tuple[float, float]]] = {}  # id → [(ts, price)]
        self.validated_pairs: list[ValidatedPair] = []
        self.last_validation_time = 0
        self.validation_interval = 300  # Re-validate every 5 minutes

    def update(self, snapshot: MarketSnapshot) -> list[dict]:
        """Add a new market snapshot and return any generated signals."""
        # Store price
        if snapshot.market_id not in self.price_history:
            self.price_history[snapshot.market_id] = []
        self.price_history[snapshot.market_id].append(
            (snapshot.timestamp, snapshot.yes_price)
        )

        # Trim old data (keep last 1000 points)
        if len(self.price_history[snapshot.market_id]) > 1000:
            self.price_history[snapshot.market_id] = \
                self.price_history[snapshot.market_id][-1000:]

        # Periodically re-detect and validate lead-lag pairs
        signals = []
        now = time.time()
        if now - self.last_validation_time > self.validation_interval:
            self._refresh_validated_pairs()
            self.last_validation_time = now

        # Generate signals for validated pairs involving this market
        for pair in self.validated_pairs:
            if pair.lag_market == snapshot.market_id:
                lead_history = self.price_history.get(pair.lead_market, [])
                if len(lead_history) >= 2:
                    recent_lead_price = lead_history[-1][1]
                    prev_lead_price = lead_history[-2][1]
                    lead_change = recent_lead_price - prev_lead_price

                    signal = self.signal_gen.generate_signal(
                        pair, lead_change, recent_lead_price
                    )
                    if signal["action"] != "hold":
                        signals.append(signal)

        return signals

    def _refresh_validated_pairs(self):
        """Re-detect and re-validate all lead-lag pairs."""
        self.validated_pairs = []
        market_ids = list(self.price_history.keys())

        for i, lead_id in enumerate(market_ids):
            for lag_id in market_ids[i + 1:]:
                lead_data = self.price_history[lead_id]
                lag_data = self.price_history[lag_id]

                if len(lead_data) < 10 or len(lag_data) < 10:
                    continue

                # Align timestamps
                lead_ts, lead_prices = zip(*lead_data)
                lag_ts, lag_prices = zip(*lag_data)

                # Detect
                candidates = self.detector.detect_lead_lag(
                    list(lead_prices), list(lag_prices),
                    list(lead_ts), lead_id, lag_id,
                )

                # Validate each candidate
                for candidate in candidates:
                    validated = self.filter.validate(
                        candidate,
                        lead_question=f"Market {lead_id}",  # Would use actual question
                        lag_question=f"Market {lag_id}",
                    )
                    if validated:
                        self.validated_pairs.append(validated)


# --- Demo / Usage ---
if __name__ == "__main__":
    print("=== LLM Semantic Lead-Lag Filter Demo ===\n")

    pipeline = LeadLagPipeline()

    # Simulate market data
    np.random.seed(42)
    base_price = 0.55
    n_points = 50

    # Market A (lead) — Fed rate decision
    lead_prices = [base_price]
    for _ in range(n_points - 1):
        change = np.random.normal(0, 0.02)
        lead_prices.append(max(0.05, min(0.95, lead_prices[-1] + change)))

    # Market B (lag) — follows lead with ~60s delay and noise
    lag_prices = [0.48]
    for i in range(n_points - 1):
        if i > 3:  # Lag of ~3 timesteps
            lead_effect = (lead_prices[i - 3] - lead_prices[i - 4]) * 0.7
        else:
            lead_effect = 0
        change = np.random.normal(0, 0.015) + lead_effect
        lag_prices.append(max(0.05, min(0.95, lag_prices[-1] + change)))

    # Feed data into pipeline
    t = time.time()
    for i in range(n_points):
        snapshot_a = MarketSnapshot(
            market_id="fed_rate_decision",
            question="Will the Fed cut rates in June 2026?",
            yes_price=lead_prices[i],
            no_price=1 - lead_prices[i],
            timestamp=t + i * 60,  # 1 minute intervals
        )
        snapshot_b = MarketSnapshot(
            market_id="mortgage_rates",
            question="Will 30-year mortgage rates fall below 6%?",
            yes_price=lag_prices[i],
            no_price=1 - lag_prices[i],
            timestamp=t + i * 60,
        )

        signals_a = pipeline.update(snapshot_a)
        signals_b = pipeline.update(snapshot_b)

        for sig in signals_a + signals_b:
            print(f"Signal: {sig['action']} on {sig['lag_market']}")
            print(f"  Position size: {sig['position_size']:.2%}")
            print(f"  Confidence: {sig['confidence']:.2f}")
            print(f"  Reason: {sig['reason']}")
            print()

    print(f"\nValidated lead-lag pairs: {len(pipeline.validated_pairs)}")
    for pair in pipeline.validated_pairs:
        print(f"  {pair.lead_market} → {pair.lag_market} "
              f"(lag={pair.lag_seconds:.0f}s, conf={pair.llm_confidence:.2f})")
