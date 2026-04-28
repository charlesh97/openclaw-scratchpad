#!/usr/bin/env python3
"""
KL-Divergence Latency Arbitrage — vega research
===============================================
Source: PolySwarm (arXiv 2604.03888, Barot & Borkhatariya, April 2026)

Concept: Derive a reference probability from external CEX data,
measure market deviation using KL divergence, and exploit the
latency lag between CEX price moves and on-chain market updates.

Key components:
  1. CEX-implied probability derivation (log-normal / Black-Scholes model)
  2. KL divergence scoring vs. Polymarket/Kalshi market prices
  3. Latency arbitrage signal generation
  4. Quarter-Kelly position sizing

Reference:
  PolySwarm: A Multi-Agent LLM Framework for Prediction Market Trading
  and Latency Arbitrage — arXiv:2604.03888 (April 2026)
"""

import math
import time
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketPrice:
    market_id: str
    question: str
    platform: str           # "polymarket" | "kalshi"
    p_mkt: float            # market-implied probability (mid price)
    bid: float
    ask: float
    end_date_iso: str       # resolution time
    asset: str = "BTC"      # underlying asset
    threshold: float = 0.0  # price threshold in market question
    volume_24h: float = 0.0


@dataclass
class LatencySignal:
    market: MarketPrice
    reference_prob: float   # P_ref from CEX model
    kl_divergence: float   # KL(P_ref || P_mkt)
    edge_per_dollar: float  # directional edge estimate
    direction: str          # "buy_market" | "sell_market" | "no_signal"
    position_size_kelly: float  # Kelly fraction
    confidence: float       # 0–1
    latency_window_ms: float # estimated opportunity window


# ---------------------------------------------------------------------------
# Probability derivation from CEX data
# ---------------------------------------------------------------------------

def derive_btc_probability(
    current_btc_price: float,
    threshold: float,
    hourly_vol: float,
    time_to_resolution_hours: float,
) -> float:
    """
    Derive P(BTC > threshold) using log-normal / Black-Scholes intuition.
    
    For a BTC hourly prediction market, we model BTC returns as log-normal:
      P(S_T > K) = N(d2)  where:
        d2 = [ln(S/K) + (μ - σ²/2) * T] / (σ * sqrt(T))
    
    For hourly markets with small T, μ ≈ 0 (driftless), simplifying to:
      d2 = [ln(S/K) - σ²/2 * T] / (σ * sqrt(T))
    
    Args:
        current_btc_price: current BTC/USD price (e.g., 104500.0)
        threshold: the price threshold in the prediction market (e.g., 105000.0)
        hourly_vol: annualized vol converted to hourly (e.g., 0.60 / sqrt(365*24))
        time_to_resolution_hours: hours until market resolution (e.g., 1.0 for hourly)
    
    Returns:
        Probability in 0–1 range
    """
    if time_to_resolution_hours <= 0:
        return 0.5  # can't estimate without time
    
    T = time_to_resolution_hours / (365 * 24)  # convert to years
    
    # Handle edge cases
    if current_btc_price <= 0 or threshold <= 0 or hourly_vol <= 0:
        return 0.5
    
    # For very short-dated options, use simplified z-score approach
    if T < 1 / (365 * 24):  # less than 1 hour
        # Simple log-return z-score
        log_moneyness = math.log(current_btc_price / threshold)
        sigma_sqrt_t = hourly_vol * math.sqrt(T)
        if sigma_sqrt_t < 1e-10:
            return 0.5
        z = log_moneyness / sigma_sqrt_t
        return _normal_cdf(z)
    
    # Standard log-normal formula
    d2 = (math.log(current_btc_price / threshold) - 0.5 * hourly_vol**2 * T) / (hourly_vol * math.sqrt(T))
    return _normal_cdf(d2)


def derive_spx_probability(
    current_spx: float,
    threshold: float,
    daily_vol: float,
    days_to_resolution: float,
) -> float:
    """
    Derive P(S&P 500 > threshold) using log-normal model.
    
    For S&P markets (daily resolution):
      - Use daily vol (σ_daily ≈ σ_annual / sqrt(252))
      - T in years = days_to_resolution / 252
    """
    if days_to_resolution <= 0:
        return 0.5
    
    T = days_to_resolution / 252.0
    daily_vol_ann = daily_vol * math.sqrt(252)  # convert daily to annual if needed
    
    if daily_vol_ann <= 0:
        return 0.5
    
    d2 = (math.log(current_spx / threshold) - 0.5 * daily_vol_ann**2 * T) / (daily_vol_ann * math.sqrt(T))
    return _normal_cdf(d2)


def _normal_cdf(z: float) -> float:
    """
    Standard normal CDF using error function approximation.
    Max absolute error ~1.5e7.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


# ---------------------------------------------------------------------------
# KL divergence scoring
# ---------------------------------------------------------------------------

def compute_kl_divergence(p_ref: float, p_mkt: float, epsilon: float = 1e-10) -> float:
    """
    Compute KL(P_ref || P_mkt) = P_ref * log(P_ref / P_mkt) + (1 - P_ref) * log((1 - P_ref) / (1 - P_mkt))
    
    This measures the "surprise" of the market probability given the reference.
    
    KL > 0: market is OVERCONFIDENT relative to reference (market prob > reference)
    KL < 0: market is UNDERCONFIDENT relative to reference (market prob < reference)
    
    Args:
        p_ref: reference probability (from CEX model)
        p_mkt: market probability (from Polymarket/Kalshi)
        epsilon: small constant to avoid log(0)
    
    Returns:
        KL divergence (can be negative)
    """
    p_ref = max(epsilon, min(1 - epsilon, p_ref))
    p_mkt = max(epsilon, min(1 - epsilon, p_mkt))
    
    kl = (p_ref * math.log(p_ref / p_mkt)
          + (1 - p_ref) * math.log((1 - p_ref) / (1 - p_mkt)))
    return kl


def compute_js_divergence(p_ref: float, p_mkt: float, epsilon: float = 1e-10) -> float:
    """
    Jensen-Shannon divergence: symmetric version of KL.
    JS = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = (P+Q)/2
    Range: [0, 1], bounded.
    """
    p_ref = max(epsilon, min(1 - epsilon, p_ref))
    p_mkt = max(epsilon, min(1 - epsilon, p_mkt))
    m = (p_ref + p_mkt) / 2
    return 0.5 * compute_kl_divergence(p_ref, m, epsilon) + 0.5 * compute_kl_divergence(p_mkt, m, epsilon)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_latency_signal(
    market: MarketPrice,
    reference_prob: float,
    fee_rate: float = 0.02,
    kl_threshold: float = 0.01,
) -> LatencySignal:
    """
    Generate a LatencyArbitrageSignal from a market price + reference probability.
    
    Args:
        market: MarketPrice object
        reference_prob: P_ref from CEX model
        fee_rate: fee rate to subtract from edge
        kl_threshold: minimum KL divergence to generate a signal
    
    Returns:
        LatencySignal with direction, edge, Kelly sizing, and confidence
    """
    kl = compute_kl_divergence(reference_prob, market.p_mkt)
    
    # Direction: market overconfident → bet against market probability
    # market prob > reference → sell YES (bet against) since market inflates probability
    # market prob < reference → buy YES (bet for) since market undervalues probability
    
    if kl > kl_threshold:
        # Market is overconfident (inflated probability) → sell YES
        direction = "sell_market"
        # Edge = how much the market overstates relative to reference, minus fees
        edge = (market.p_mkt - reference_prob) - fee_rate
    elif kl < -kl_threshold:
        # Market is underconfident (deflated probability) → buy YES
        direction = "buy_market"
        edge = (reference_prob - market.p_mkt) - fee_rate
    else:
        direction = "no_signal"
        edge = 0.0
    
    # Quarter-Kelly sizing: f* = bp - q / b, capped at 0.25 (quarter Kelly)
    # Simplified: edge / (1 - p_mkt) for directional bets
    if edge > 0 and direction != "no_signal":
        kelly = _quarter_kelly(edge, market.p_mkt if direction == "sell_market" else reference_prob)
    else:
        kelly = 0.0
    
    # Confidence scales with KL magnitude and market liquidity
    liquidity_score = min(1.0, market.volume_24h / 10000)  # rough heuristic
    kl_magnitude = abs(kl)
    confidence = min(0.95, 0.5 + kl_magnitude * 10) * (0.7 + 0.3 * liquidity_score)
    
    # Latency window estimate (from IMDEA 2026 data: avg 2.7s, sub-100ms for top bots)
    if kl_magnitude > 0.05:
        latency_window_ms = 500  # larger edge = wider window
    elif kl_magnitude > 0.02:
        latency_window_ms = 1500
    else:
        latency_window_ms = 2700
    
    return LatencySignal(
        market=market,
        reference_prob=reference_prob,
        kl_divergence=kl,
        edge_per_dollar=edge,
        direction=direction,
        position_size_kelly=kelly,
        confidence=confidence,
        latency_window_ms=latency_window_ms,
    )


def _quarter_kelly(edge: float, probability: float) -> float:
    """
    Quarter-Kelly sizing: f = edge / probability, capped at 0.25.
    
    Simplified Kelly for binary outcome:
      f* ≈ edge / probability_of_outcome
    
    Quarter-Kelly reduces volatility while capturing ~75% of Kelly's growth.
    """
    if probability <= 0 or probability >= 1:
        return 0.0
    kelly = edge / probability
    return max(0.0, min(0.25, kelly))


# ---------------------------------------------------------------------------
# Full scanner
# ---------------------------------------------------------------------------

def scan_markets(
    markets: list[MarketPrice],
    btc_price: float,
    btc_hourly_vol: float,
    fee_rate: float = 0.02,
    kl_threshold: float = 0.01,
    min_edge: float = 0.005,
) -> list[LatencySignal]:
    """
    Main entry point: scan a list of markets for latency arb signals.
    
    Args:
        markets: List of MarketPrice objects
        btc_price: Current BTC/USD price
        btc_hourly_vol: BTC hourly realized volatility (annualized / sqrt(365*24))
        fee_rate: fee rate to subtract from edge
        kl_threshold: minimum |KL| to generate signal
        min_edge: minimum edge to include in results
    """
    signals = []
    
    for m in markets:
        # Derive reference probability based on asset type
        if m.asset.upper() in ("BTC", "BITCOIN"):
            time_hours = _hours_until(m.end_date_iso)
            p_ref = derive_btc_probability(
                current_btc_price=btc_price,
                threshold=m.threshold,
                hourly_vol=btc_hourly_vol,
                time_to_resolution_hours=time_hours,
            )
        elif m.asset.upper() in ("SPX", "S&P", "SP500"):
            days = _days_until(m.end_date_iso)
            # Assume 15% annual vol for SPX
            p_ref = derive_spx_probability(
                current_spx=btc_price,  # reused for SPX price
                threshold=m.threshold,
                daily_vol=0.15,
                days_to_resolution=days,
            )
        else:
            # Fallback: use simple momentum model
            p_ref = m.p_mkt  # no edge
        
        signal = generate_latency_signal(m, p_ref, fee_rate, kl_threshold)
        
        if signal.direction != "no_signal" and signal.edge_per_dollar >= min_edge:
            signals.append(signal)
    
    # Sort by edge descending
    signals.sort(key=lambda s: s.edge_per_dollar, reverse=True)
    return signals


def _hours_until(iso_date: str) -> float:
    """Hours until an ISO date from now."""
    try:
        future = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        delta = future - datetime.now().astimezone()
        return max(0.0, delta.total_seconds() / 3600)
    except Exception:
        return 24.0  # default assumption


def _days_until(iso_date: str) -> float:
    """Days until an ISO date from now."""
    return _hours_until(iso_date) / 24


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_signal(signal: LatencySignal) -> str:
    m = signal.market
    edge_pct = signal.edge_per_dollar * 100
    kelly_pct = signal.position_size_kelly * 100
    kl_sign = "+" if signal.kl_divergence > 0 else ""
    
    return f"""
⚡ KL LATENCY ARB — {signal.direction.replace("_", " ").upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Market:     {m.question[:70]}
Platform:   {m.platform} | Vol 24h: ${m.volume_24h:,.0f}
Threshold:  ${m.threshold:,.0f} | BTC price: ${m.asset}

P_ref (CEX model):  {signal.reference_prob:.4f}
P_mkt (market):      {m.p_mkt:.4f}
KL divergence:       {kl_sign}{signal.kl_divergence:.4f}
Edge:                {edge_pct:.3f}% per dollar
Kelly fraction:     {kelly_pct:.1f}% of bankroll
Latency window:     ~{signal.latency_window_ms:.0f}ms
Confidence:         {signal.confidence:.0%}
""".strip()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo():
    """Run demo with synthetic market data."""
    
    btc_price = 104500.0
    # BTC hourly vol: ~60% annualized → hourly = 0.60 / sqrt(8760) ≈ 0.0064
    btc_hourly_vol = 0.60 / math.sqrt(365 * 24)
    
    markets = [
        # BTC 1-hour market: PM says 52%, CEX model says 55%
        MarketPrice(
            market_id="poly_btc_hour_1",
            question="Will BTC be above $105,000 by 4pm ET today?",
            platform="polymarket",
            p_mkt=0.52,
            bid=0.51, ask=0.53,
            end_date_iso="2026-04-28T20:00:00Z",
            asset="BTC",
            threshold=105000.0,
            volume_24h=25000.0,
        ),
        # BTC 1-hour market: PM says 48%, CEX model says 43% → overconfident PM
        MarketPrice(
            market_id="kalshi_btc_hour_2",
            question="Will BTC be above $106,000 by 5pm ET today?",
            platform="kalshi",
            p_mkt=0.48,
            bid=0.47, ask=0.49,
            end_date_iso="2026-04-28T21:00:00Z",
            asset="BTC",
            threshold=106000.0,
            volume_24h=15000.0,
        ),
        # Low-volume market: should be deweighted
        MarketPrice(
            market_id="poly_btc_hour_3",
            question="Will BTC be above $107,000 by 6pm ET today?",
            platform="polymarket",
            p_mkt=0.40,
            bid=0.39, ask=0.41,
            end_date_iso="2026-04-28T22:00:00Z",
            asset="BTC",
            threshold=107000.0,
            volume_24h=500.0,
        ),
    ]
    
    print("=" * 65)
    print("KL DIVERGENCE LATENCY ARBITRAGE — DEMO")
    print(f"BTC price: ${btc_price:,.0f} | Hourly vol: {btc_hourly_vol:.5f}")
    print("=" * 65)
    
    signals = scan_markets(markets, btc_price, btc_hourly_vol,
                           fee_rate=0.02, kl_threshold=0.01, min_edge=0.005)
    
    if not signals:
        print("\nNo signals above threshold.")
        return
    
    for sig in signals:
        print(format_signal(sig))
        print()


if __name__ == "__main__":
    _demo()
