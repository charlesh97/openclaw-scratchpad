# Synthetic Probability from BTC Options Implied Volatility

**Status:** YES — High Confidence, Low Complexity  
**Source:** Moontower substack; Deribit BTC options data  
**Research date:** 2026-04-27 by vega

---

## TL;DR

Use BTC options market implied volatility (IV) to compute a **market-independent probability estimate** for BTC prediction markets. Compare your estimate to Polymarket/Kalshi prices — deviation from IV-derived probability = edge signal. Low implementation cost (~10 lines of Python), applicable to all BTC-duration markets.

---

## The Core Formula

The Black-Scholes z-score approach converts options IV into a fair probability:

```
z = (X - F) / (σ × √T)

fair_probability = Φ(z)   # Φ = standard normal CDF
```

Where:
- `F` = current BTC price (or options-implied forward)
- `X` = strike price (the prediction market threshold)
- `σ` = annualized implied volatility from BTC options (Deribit)
- `T` = time to market resolution in years
- `Φ` = cumulative standard normal distribution

The output is the probability that BTC > X at expiry — compare to market price to find edge.

---

## Worked Example

BTC at $89,000. BTC options show 55% annualized IV.  
Kalshi market: "Will BTC close above $100,000 by Dec 31, 2026" — currently priced at 65¢ YES.

```
F  = 89,000
X  = 100,000
σ  = 0.55
T  = (Dec 31 - today) / 365 ≈ 0.67 years

z  = (100,000 - 89,000) / (0.55 × √0.67)
   = 11,000 / (0.55 × 0.82)
   = 11,000 / 0.451
   = 24,390  ← extremely high z-score
   Φ(24,390) ≈ 1.0 (essentially 100%)

vs. market price of 65¢ → 35¢ overestimate by market
→ STRONG SELL signal on the YES
```

In practice, IV changes continuously — recompute daily or intraday.

---

## Sample Code

```python
from scipy.stats import norm
import math

def btc_options_probability(
    spot_btc: float,           # current BTC/USD price
    strike: float,             # prediction market threshold (e.g. 100_000)
    iv: float,                 # annualized implied vol from BTC options (e.g. 0.55)
    duration_hours: float,     # hours until market resolution
) -> float:
    """
    Compute fair probability BTC > strike using Black-Scholes z-score.

    Returns: probability (0.0–1.0) that BTC settles above strike.
    """
    T = duration_hours / (24 * 365)  # convert to years
    sigma_times_sqrtT = iv * math.sqrt(T)

    if sigma_times_sqrtT == 0:
        return 1.0 if spot_btc >= strike else 0.0

    z = (strike - spot_btc) / (sigma_times_sqrtT * spot_btc)
    # Actually use log-return form for more accuracy:
    # z = ln(F/K) / (σ × √T)
    z = math.log(spot_btc / strike) / sigma_times_sqrtT

    # P(BTC > strike) = 1 - Φ(z)  [since z is below strike → positive]
    prob = norm.cdf(-z)  # flip because we're asking above
    return prob


def find_edge(
    market_price: float,        # YES price on Polymarket/Kalshi ($0.00–$1.00)
    estimated_probability: float,
    maker_fee: float = 0.02,
    taker_fee: float = 0.03,
) -> dict:
    """
    Compute edge: estimated_probability vs market-implied probability,
    adjusted for fees.
    """
    market_implied_prob = market_price * (1 - taker_fee)
    gross_edge = estimated_probability - market_implied_prob
    net_edge = gross_edge - maker_fee

    return {
        "market_price": market_price,
        "market_implied_prob": market_implied_prob,
        "estimated_probability": estimated_probability,
        "gross_edge": gross_edge,
        "net_edge": net_edge,
        "signal": "BUY YES" if net_edge > 0.02 else "HOLD",
    }
```

---

## Data Sources

| Source | What's needed | Access |
|---|---|---|
| Deribit BTC options | ATM IV for expiries matching market durations | Public API (no auth) |
| Coinbase | BTC/USD spot price | Public API |
| Kaiko | Historical IV data for backtesting | Free tier available |

Deribit public endpoints:
- `https://www.deribit.com/api/v2/public/get_instruments?currency=BTC` — list options
- `https://www.deribit.com/api/v2/public/get_volatility_index?currency=BTC` — IV index

---

## Key Parameters

| Parameter | Notes |
|---|---|
| `iv` source | Use nearest-expiry ATM IV matching the market duration |
| `duration_hours` | Must match the prediction market's resolution window |
| `maker_fee` | ~2% on Kalshi; ~0% on Polymarket (adjust accordingly) |
| `taker_fee` | ~3% on Kalshi; ~2% on Polymarket |

---

## Key Risks

| Risk | Mitigation |
|---|---|
| IV is forward-looking (embeds expectations) | Use at your own discretion — IV reflects market consensus |
| Options market is thin for long durations | Prefer short-duration markets (1hr–1week) where Deribit liquidity is deep |
| Model misspecification (GBM assumption) | Treat as one signal among several; don't use in isolation |

---

## Next Steps

1. Pull live BTC spot + Deribit IV data via public API
2. Backtest on historical BTC prediction markets (Nov–Dec 2025)
3. Compare z-score probability to actual market resolution prices
4. Add to alert queue as probabilistic signal layer

---

## References

- [Moontower: arbitrage using option chains](https://navnoorbawa.substack.com/p/arbitrage-evolution-from-morgan-stanleys)
- [Deribit public API docs](https://docs.deribit.com/)
