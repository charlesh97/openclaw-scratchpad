"""
Synthetic BTC Probability from Options Implied Volatility
=========================================================
Implements Finding 6 from 2026-04-27 research.

Derives an independent "fair" probability that BTC will be above a given
strike price at a given expiry, using BTC options market implied volatility.
Compares this to Polymarket/Kalshi prices to detect edge.

Formula:
    z = (X - F) / (σ * sqrt(T))
    p = N(z)  # probability BTC > X

Where:
    X  = strike price
    F  = forward price (~current BTC spot or computed from term structure)
    σ  = annualized implied volatility from BTC options chain
    T  = time to expiry in years
    N  = standard normal CDF

Reference:
    - Moontower substack (navnoorbawa.substack.com)
    - Flashbots research on prediction market vs. options-implied probabilities
    - arXiv:2508.03474 (Saguillo et al.)

Usage:
    p = btc_options_probability(strike=100_000, T_hours=168, iv=0.65)
    # p ≈ probability BTC > $100k in next 7 days per options market
    # If Polymarket shows $0.58 for YES on "BTC>$100k by Friday", edge = 0.58 - p
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone


# ── Standard normal CDF (Abramowitz & Stegun approximation) ────────────────────
def _norm_cdf(z: float) -> float:
    """Standard normal CDF via error function approximation."""
    # Constants for rational approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = -1 if z < 0 else 1
    z = abs(z) / math.sqrt(2)
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)
    return 0.5 * (1.0 + sign * y)


def btc_options_probability(
    strike_usd: float,
    T_hours: float,
    iv: float,
    spot_usd: Optional[float] = None,
    forward_usd: Optional[float] = None,
) -> dict:
    """
    Compute probability that BTC spot > strike at expiry, from options IV.

    Args:
        strike_usd   : strike price in USD
        T_hours      : time to expiry in hours
        iv           : annualized implied volatility (e.g., 0.65 for 65%)
        spot_usd     : current BTC spot price (fallback if no forward)
        forward_usd  : explicit forward price (preferred if available)

    Returns:
        dict with keys: z_score, probability, edge_signal, interpretation

    The probability is the risk-neutral probability under the assumption that
    BTC follows a log-normal process (standard options pricing assumption).
    This is the same probability concept used in Deribit/Bitmex BTC options.
    """
    T_years = T_hours / (24.0 * 365.0)

    # Avoid division by zero for very short expiries
    if T_years < 1e-6:
        raise ValueError(f"T to expiry too small: {T_hours} hours")

    # Forward price: use provided forward or approximate with spot
    if forward_usd is None:
        if spot_usd is None:
            raise ValueError("Must provide either forward_usd or spot_usd")
        forward_usd = spot_usd  # Simplification: no cost-of-carry for BTC

    # Z-score: (X - F) / (σ * sqrt(T))
    denom = iv * math.sqrt(T_years)
    z_score = (strike_usd - forward_usd) / (forward_usd * denom) if forward_usd else 0.0

    # Probability BTC > strike (one-tailed)
    probability = _norm_cdf(z_score)

    # Edge signal: deviation from model probability
    # Not stored here — compute externally as: market_price - probability
    edge_signal = None  # computed externally where market price is known

    # Interpretation
    if probability > 0.9:
        interpretation = "Very likely (>90%)"
    elif probability > 0.7:
        interpretation = "Likely (70-90%)"
    elif probability > 0.5:
        interpretation = "Lean (>50%)"
    elif probability > 0.3:
        interpretation = "Unlikely (30-50%)"
    else:
        interpretation = "Very unlikely (<30%)"

    return {
        "strike_usd": strike_usd,
        "forward_usd": forward_usd,
        "iv": iv,
        "T_hours": T_hours,
        "T_years": T_years,
        "z_score": z_score,
        "probability": probability,
        "interpretation": interpretation,
    }


def compute_edge(market_yes_price: float, probability: float) -> dict:
    """
    Compute edge signal between market price and model probability.

    Args:
        market_yes_price : market price for YES (0.0 to 1.0)
        probability      : model-derived probability from options IV

    Returns:
        dict with edge, edge_pct, annualized_return (if edge exists)
    """
    edge = market_yes_price - probability
    edge_pct = abs(edge) / probability if probability > 1e-6 else 0.0

    # Rough annualized return estimate (assuming 50% win rate on symmetric bets)
    # This is indicative only — real P&L depends on strike resolution
    breakeven_edge = 0.0  # edge must be positive to be profitable after fees
    is_profitable = edge > 0.02  # 2% minimum after fees/slippage buffer

    return {
        "edge": edge,
        "edge_pct": edge_pct,
        "market_yes_price": market_yes_price,
        "model_probability": probability,
        "is_edge": edge > 0.01,       # 1% minimum edge threshold
        "is_strong_edge": edge > 0.03,  # 3% strong edge
        "profitable_after_fees": edge > 0.02,
    }


# ── Data fetching helpers ─────────────────────────────────────────────────────

@dataclass
class DeribitIVQuote:
    """Represents a BTC options IV quote from Deribit."""
    strike: float
    expiry: str          # ISO timestamp
    iv_bid: float       # bid IV (annualized)
    iv_ask: float       # ask IV (annualized)
    forward_price: float

    @property
    def iv_mid(self) -> float:
        return (self.iv_bid + self.iv_ask) / 2.0


def fetch_atm_btc_iv(deribit_instrument: str = "BTC-29MAY26") -> dict:
    """
    Fetch ATM (at-the-money) IV for a BTC options expiry from Deribit.

    Uses Deribit's public API (no auth required for market data).

    Returns dict with strike, T_hours, iv, forward.

    Note: In production, use the official Deribit Python SDK or REST API.
    This stub demonstrates the API shape.
    """
    # Deribit public API endpoint for BTC options quotes
    # GET https://www.deribit.com/api/v2/public/get_book_summary_by_instrument
    # ?instrument_name=BTC-29MAY26-100000-C  (call option, strike 100k)
    #
    # For IV estimation: use OTM options ATM skew or fetch via:
    # GET https://www.deribit.com/api/v2/public/get_volatility
    #
    # Real implementation would use: requests.get(...)
    # Stub returns example values — replace with live API call in production.
    return {
        "instrument": deribit_instrument,
        "strike": 100_000.0,
        "expiry": "2026-05-29T08:00:00Z",
        "iv": 0.62,          # 62% annualized IV
        "forward": 105_000.0,  # forward price implied by put-call parity
        "T_hours": 768.0,     # ~32 days
    }


def estimate_iv_from_options_chain(
    otm_strike: float,
    atm_strike: float,
    otm_iv: float,
    atm_iv: float,
) -> float:
    """
    Interpolate/extrapolate ATM IV from nearby strikes.

    BTC options markets are often quoted with ATM IV and skew parameters.
    Use this when you only have OTM quotes and need ATM.
    """
    # Simple linear interpolation — good enough for strikes within 10%
    if atm_strike == otm_strike:
        return otm_iv
    # Weight by proximity to ATM
    w = 1.0 - abs(otm_strike - atm_strike) / atm_strike
    w = max(0.0, min(1.0, w))
    return w * atm_iv + (1.0 - w) * otm_iv


# ── Demo / backtest helper ────────────────────────────────────────────────────

def demo():
    """Print sample probability outputs for various strikes/expiries."""
    print("=== BTC Options Implied Probability ===\n")

    spot = 105_000.0
    scenarios = [
        # (strike, T_hours, iv)
        (100_000, 168, 0.60),   # BTC > $100k in 7 days at 60% IV
        (110_000, 168, 0.65),   # BTC > $110k in 7 days
        (100_000, 720, 0.55),   # BTC > $100k in 30 days
        (120_000, 168, 0.70),   # BTC > $120k in 7 days (OTM)
        (100_000, 24,  0.80),   # BTC > $100k in 24h (short expiry)
    ]

    for strike, T, iv in scenarios:
        result = btc_options_probability(strike, T, iv, spot_usd=spot)
        print(f"Strike ${strike:,.0f}  T={T:4.0f}h  IV={iv:.0%}")
        print(f"  z-score: {result['z_score']:.3f}")
        print(f"  P(BTC>${strike:,.0f}): {result['probability']:.1%}  [{result['interpretation']}]")
        # Compare to hypothetical market price
        market_price = 0.50
        edge = compute_edge(market_price, result['probability'])
        if edge['is_edge']:
            print(f"  ⚠️  Edge vs market@${market_price:.2f}: {edge['edge']:+.1%} {'✅' if edge['profitable_after_fees'] else '⚠️ fees'}")
        print()


if __name__ == "__main__":
    demo()