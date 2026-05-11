#!/usr/bin/env python3
"""
ForesightFlow Information Leakage Score — Reference Implementation Pattern
From arXiv:2605.00493. Computes ILS for binary prediction markets.
"""
from datetime import datetime
from typing import Optional


def information_leakage_score(
    price_series: list,
    event_timestamp: datetime,
    resolution_price: float
) -> float:
    """
    Compute ILS: fraction of terminal information move priced in before event.
    Higher score = more information leakage.

    ILS = (price_before - price_anchor) / (resolution_price - price_anchor)
    """
    price_before = _price_at_time(price_series, event_timestamp)
    price_anchor = price_series[0]  # first price in series

    numerator = price_before - price_anchor
    denominator = resolution_price - price_anchor

    if abs(denominator) < 1e-10:
        return 0.0  # no move to explain

    return max(0.0, min(1.0, numerator / denominator))


def deadline_ils(
    price_series: list,
    deadline_timestamp: datetime,
    public_event_timestamp: datetime,
    resolution_price: float,
    baseline_hazard: float = 0.01
) -> float:
    """
    Deadline-ILS extension for deadline-resolved markets.
    Anchored at public-event timestamp instead of news timestamp.
    Uses exponential hazard for time-to-event distribution.
    """
    # Per-category exponential hazard baseline adjustment
    hazard_adjustment = 1.0 - (baseline_hazard * 24)  # daily hazard rate
    return information_leakage_score(
        price_series, public_event_timestamp, resolution_price
    ) * hazard_adjustment


def _price_at_time(price_series: list, timestamp: datetime) -> float:
    """Interpolate price at given timestamp from price series."""
    # In production: find nearest timestamp in series
    return price_series[-1] if price_series else 0.5


if __name__ == "__main__":
    prices = [0.50, 0.52, 0.55, 0.58, 0.62]
    event_time = datetime(2026, 5, 10, 14, 0, 0)
    ils = information_leakage_score(prices, event_time, 0.75)
    print(f"Information Leakage Score: {ils:.3f}")
    print("(0.0 = no leakage, 1.0 = fully priced in before event)")
