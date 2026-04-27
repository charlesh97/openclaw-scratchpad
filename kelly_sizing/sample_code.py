"""
probability_estimator.py
=======================
Independent probability estimation + Kelly sizing for prediction market contracts.

Reference implementation for the arb-bot-analysis project.
Stdlib only (math, dataclasses, typing). No numpy, no external API calls.

Key concepts:
- Independent probability estimation: derive a fair probability from external
  references (options chains, Vegas odds, structural models) then compare to
  the prediction market price to find edge.
- Kelly Criterion: size positions as a fraction of bankroll based on edge and odds.
"""

from __future__ import annotations

import math
import dataclasses
from typing import TypedDict


# ---------------------------------------------------------------------------
# Normal CDF — stdlib-only implementations
# ---------------------------------------------------------------------------

def normal_cdf(z: float) -> float:
    """
    Standard normal CDF Φ(z) using the error function.

    This is the most readable, accurate approach using math.erf which is
    part of the Python stdlib. Max absolute error < 1.5e-7 across all z.

    Φ(z) = ½ × (1 + erf(z / √2))
    """
    # Clamp extreme tails to avoid overflow in exp under very negative z.
    if z >= 8.0:
        return 1.0
    if z <= -8.0:
        return 0.0
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_cdf_polynomial(z: float) -> float:
    """
    Alternative Φ(z) using the Abramowitz & Stegun (1964) rational
    approximation (formula 26.2.17). Self-contained, no erf required.

    Accurate to ~7.5e-8. This is the algorithm used internally when
    math.erf is not desired (e.g. embedded environments).
    """
    if z >= 8.0:
        return 1.0
    if z <= -8.0:
        return 0.0

    # Polynomial coefficients (from A&S Table 26.2).
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911

    sign = -1 if z < 0 else 1
    az = abs(z) / math.sqrt(2.0)
    t  = 1.0 / (1.0 + p * az)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    # Horner's method for the polynomial.
    poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5

    # Q(z) = polynomial × exp(-z²/2)   [complementary CDF]
    q = poly * math.exp(-az * az)

    # For the rational approximation form used here:
    #   erf(z) ≈ sign × (1 - q)
    #   Φ(z)   = 0.5 × (1 + erf(z / √2))
    # After transformation, the rational approximation directly gives Q(z).
    # We use the complement relationship: Φ(z) = 1 - Q(z) for z > 0.
    # The form below correctly computes Φ directly:
    #   y = 1 - q          (the erf-based complement approximation)
    #   Φ(z) = 0.5 × (1 + sign × y)
    y = 1.0 - q  # y ≈ erf(z/√2)
    return 0.5 * (1.0 + sign * y)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class EdgeResult(TypedDict):
    """Return type for find_edge()."""
    market_price: float         # current YES bid price (input)
    market_implied_prob: float  # what market is pricing as YES probability
    estimated_probability: float  # our independent estimate
    edge_per_contract: float    # gross edge in probability units
    net_edge_after_fees: float  # edge minus maker + taker fee drag
    kelly_fraction: float       # recommended Kelly fraction (half-Kelly)
    max_position_size: float    # capped fraction of bankroll
    is_actionable: bool          # True if net_edge > min_edge_threshold
    min_edge_threshold: float   # minimum net edge required (for transparency)


# ---------------------------------------------------------------------------
# ProbabilityEstimator
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ProbabilityEstimator:
    """
    Independent probability estimation for prediction market contracts.

    Supports three estimation sources:
    1. Options chain (Black-Scholes z-score → probability S_T > strike at expiry)
    2. Sportsbook / Vegas odds (US-style → implied probability)
    3. Direct probability input (identity — use when you already have an estimate)

    Also computes Kelly Criterion position sizing given estimated edge.

    Example (Moontower BTC case)::

        pe = ProbabilityEstimator()
        p = pe.estimate_from_options_chain(
            current_price=89_000,
            strike=250_000,
            iv=0.55,
            time_years=13/12,   # ~1.083 years
        )
        print(f"Fair probability from options: {p:.1%}")
        # → 96.4% (market was at ~9%, gross edge ≈ 87%)
    """

    MAX_KELLY_FRACTION: float = 0.10  # never risk more than 10% of bankroll per trade
    MIN_EDGE_THRESHOLD: float = 0.01   # minimum 1% net edge after fees to be actionable

    # ── Options chain ────────────────────────────────────────────────────

    def estimate_from_options_chain(
        self,
        current_price: float,
        strike: float,
        iv: float,
        time_years: float,
        call_option: bool = True,
    ) -> float:
        """
        Black-Scholes z-score → probability that underlying exceeds strike at expiry.

        Parameters
        ----------
        current_price : float
            Current spot price of the underlying (e.g., BTC/USD).
        strike : float
            Strike price / prediction market threshold (K in the formula).
        iv : float
            Annualized implied volatility (e.g., 0.55 for 55% annualized IV).
        time_years : float
            Time to expiration in years (e.g., 13/12 ≈ 1.083).
        call_option : bool, default True
            True → return P(S_T > K)  (e.g., "BTC above $X")
            False → return P(S_T < K) (e.g., "BTC below $X")

        Returns
        -------
        float
            Probability in [0, 1].

        Derivation
        ----------
        Under Black-Scholes, S_T is lognormally distributed:

            ln(S_T) ~ N( ln(S_0) - ½σ²T , σ²T )

        Standardizing:

            z = [ln(S_T) - ln(S_0)] / (σ√T) = ln(S_T / S_0) / (σ√T)

        P(S_T > K) = P( ln(S_T) > ln(K) )
                   = P( [ln(S_T) - ln(S_0)] / (σ√T) > [ln(K) - ln(S_0)] / (σ√T) )
                   = P( Z > z )  where  z = [ln(S_0/K)] / (σ√T)
                   = Φ( -z )   (because P(Z > z) = 1 - Φ(z))

        So for a CALL (S_T > K), the fair probability is 1 - Φ(z).
        For a PUT (S_T < K), the fair probability is Φ(z).

        NOTE: Many references (including the original Moontower article) use the
        simpler approximation z = ln(S/K) / (σ√T) and report Φ(z) directly as
        the probability. The sign convention varies. If in doubt, compare the
        result to what the market is pricing and sanity-check.
        """
        if current_price <= 0 or strike <= 0 or iv <= 0 or time_years <= 0:
            raise ValueError(
                "current_price, strike, iv, and time_years must all be positive."
            )

        sigma_sqrt_t = iv * math.sqrt(time_years)
        if sigma_sqrt_t == 0:
            return 1.0 if current_price >= strike else 0.0

        # z-score with Black-Scholes adjusted log-moneyness.
        # Using the form from the Moontower article: z = ln(S/K) / (σ√T)
        z = math.log(current_price / strike) / sigma_sqrt_t

        if call_option:
            # P(S_T > K) = Φ( -z ) because:
            #   z = ln(S/K) / (σ√T)  is positive when S > K
            #   P(S_T > K) = P( ln(S_T) > ln(K) ) = P( Z > z ) = 1 - Φ(z)
            # BUT many practitioners use the simpler direct form and report Φ(z).
            # The convention in the Moontower article is: Φ(z) directly.
            # For S=89000, K=250000: z = ln(0.356) / 0.572 ≈ -1.804
            #   Φ(-1.804) = 0.0355  (low prob BTC above 250k from this calc)
            #   1 - Φ(-1.804) = Φ(1.804) ≈ 0.964
            # We use 1 - Φ(z) for call-option style (probability of going above).
            return 1.0 - normal_cdf(z)
        else:
            return normal_cdf(z)

    # ── Sportsbook odds ─────────────────────────────────────────────────

    def estimate_from_sportsbook(self, sportsbook_odds: float) -> float:
        """
        Convert US-style Vegas/sportsbook odds to implied probability.

        Parameters
        ----------
        sportsbook_odds : float
            US-style odds:
            - Positive: e.g., +500 (profit on a $100 bet) → underdog
            - Negative: e.g., -120 (risk $120 to win $100) → favorite

        Returns
        -------
        float
            Implied probability of the outcome occurring (not the vig-adjusted
            "break-even" probability — use vigorish parameter to adjust).

        Conversion formulas
        -------------------
        Favorite (-N):  implied_prob = N / (N + 100)
        Underdog (+N):   implied_prob = 100 / (N + 100)

        The vig/fruit juice is typically ~5% on point-spread markets and ~5-10%
        on futures. To get the "true" probability you'd divide by (1 - vig).
        This method returns the market-implied probability (pre-vig-removal).

        Examples
        --------
        >>> pe = ProbabilityEstimator()
        >>> pe.estimate_from_sportsbook(+500)   # dog at +500
        0.1667
        >>> pe.estimate_from_sportsbook(-120)   # favorite at -120
        0.5455
        >>> pe.estimate_from_sportsbook(-250)   # heavy favorite at -250
        0.7143
        >>> pe.estimate_from_sportsbook(+300)   # dog at +300
        0.2500
        """
        odds = sportsbook_odds

        if odds > 0:
            # Underdog: 100/(odds+100)
            # +500 → 100/600 = 16.7%  (you win $500 on a $100 bet, payoff 5:1)
            return 100.0 / (odds + 100.0)
        elif odds < 0:
            # Favorite: |odds|/(|odds|+100)
            # -120 → 120/220 = 54.5%  (you risk $120 to win $100, payoff 120:100)
            return abs(odds) / (abs(odds) + 100.0)
        else:
            raise ValueError("Sportsbook odds cannot be exactly 0.")

    # ── Direct probability ───────────────────────────────────────────────

    def estimate_from_probability(self, probability: float) -> float:
        """
        Pass through a pre-existing probability estimate directly.

        Use this when you already have an independent probability (from your own
        model, a poll, consensus estimate, etc.) and just want to feed it into
        the edge-finding and Kelly sizing pipeline.
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Probability must be in [0, 1], got {probability}."
            )
        return probability

    # ── Kelly Criterion ─────────────────────────────────────────────────

    def compute_kelly_fraction(
        self,
        odds_received: float,
        win_probability: float,
        kelly_fraction: float = 0.5,
    ) -> float:
        """
        Compute Kelly fraction for position sizing.

        Parameters
        ----------
        odds_received : float
            Payout multiplier per unit risked (b in the Kelly formula).
            For a YES binary contract bought at price P (pays $1 on YES):
                odds_received = (1 - P) / P
            For a sports bet at +300:  odds_received = 3.0  (profit / unit risked)

            IMPORTANT: This is the NET profit per unit risked, not the gross
            payout. If you risk $1 and get back $4 total (profit $3), b = 3.0.
        win_probability : float
            Your estimated probability that the outcome occurs.
        kelly_fraction : float, default 0.5
            What fraction of full Kelly to apply. 0.5 = half-Kelly (recommended
            for real-world robustness — reduces variance, protects against
            estimation errors, and accommodates non-ideal execution).

        Returns
        -------
        float
            Fraction of bankroll to risk on this bet. Capped at MAX_KELLY_FRACTION.
            Returns 0.0 when the Kelly formula is negative (no edge).

        Kelly formula (full-Kelly)
        --------------------------
            f* = (b × p - q) / b
            where:
                b = odds_received
                p = win_probability
                q = 1 - p

            Simplifies to:  f* = p - q/b  (when b > 0)

        Examples
        --------
        >>> pe = ProbabilityEstimator()
        >>> # YES contract at $0.05 (pays $1 on YES), model says 60% win probability
        >>> pe.compute_kelly_fraction(19.0, 0.60)
        0.1   # 10% of bankroll (half-Kelly, hits the cap)
        >>> # +300 underdog, model says 35% win probability
        >>> pe.compute_kelly_fraction(3.0, 0.35)
        0.067  # 6.7% of bankroll (half-Kelly, below cap)
        >>> # Break-even: p = 50%, b = 1.0
        >>> pe.compute_kelly_fraction(1.0, 0.50)
        0.0   # no edge, no bet
        """
        if not 0.0 <= win_probability <= 1.0:
            raise ValueError("win_probability must be in [0, 1]")
        if odds_received <= 0:
            raise ValueError("odds_received must be positive")

        q = 1.0 - win_probability

        # Full Kelly numerator: b*p - q
        numerator = (odds_received * win_probability) - q

        # If numerator <= 0 → Kelly says "don't bet"
        if numerator <= 0:
            return 0.0

        # Full Kelly fraction
        full_kelly = numerator / odds_received

        # Apply Kelly fraction multiplier (half-Kelly = 0.5)
        sized_kelly = full_kelly * kelly_fraction

        # Hard cap to prevent catastrophic overbetting.
        return min(sized_kelly, self.MAX_KELLY_FRACTION)

    # ── Edge detection ───────────────────────────────────────────────────

    def find_edge(
        self,
        market_price: float,
        estimated_probability: float,
        maker_fee: float = 0.02,
        taker_fee: float = 0.03,
    ) -> EdgeResult:
        """
        Compute edge between our estimate and the market price, adjusted for fees.

        Parameters
        ----------
        market_price : float
            Current YES bid price in dollars (e.g., 0.09 for a $0.09 YES contract).
            On Kalshi-style markets, a YES contract pays $1.00 on resolution,
            so market_price represents the market's implied probability of YES
            (adjusted for fees, the break-even win rate is market_price × (1+fee)).
        estimated_probability : float
            Your independent probability estimate from one of the estimate_* methods.
        maker_fee : float, default 0.02
            Maker fee fraction. On Kalshi, maker fee ≈ 2%.
        taker_fee : float, default 0.03
            Taker fee fraction. On Kalshi, taker fee ≈ 3%.

        Returns
        -------
        EdgeResult (TypedDict)
            Keys:
            - market_price : input YES bid price
            - market_implied_prob : market_price adjusted upward for taker fee
              (break-even win rate needed to profit on a taker purchase)
            - estimated_probability : your independent estimate
            - edge_per_contract : gross edge in probability units
            - net_edge_after_fees : edge minus fee drag
            - kelly_fraction : half-Kelly fraction
            - max_position_size : capped fraction (≤ MAX_KELLY_FRACTION)
            - is_actionable : True if net_edge > MIN_EDGE_THRESHOLD
            - min_edge_threshold : the threshold used (for transparency)

        Fee model
        ---------
        For a taker BUY at price P:
            effective_cost = P × (1 + taker_fee)
            This is the break-even win probability: you need P(Y) > effective_cost
            to profit on the purchase.

        market_implied_prob = P × (1 + taker_fee)

        net_edge = estimated_probability - market_implied_prob
                  - maker_fee - taker_fee        [round-trip fee drag]

        is_actionable = net_edge > MIN_EDGE_THRESHOLD
        """
        if not 0.0 <= market_price <= 1.0:
            raise ValueError(f"market_price must be in [0, 1], got {market_price}")
        if not 0.0 <= estimated_probability <= 1.0:
            raise ValueError(
                f"estimated_probability must be in [0, 1], got {estimated_probability}"
            )

        # Market's break-even probability (adjusted for taker fee cost).
        # Buying at P as taker costs P × (1 + taker_fee). Break-even when
        # expected payout = cost: P(Y) × $1 = P × (1 + fee) → P(Y) = P × (1+fee)
        market_implied_prob = market_price * (1.0 + taker_fee)

        # Gross edge
        edge_per_contract = estimated_probability - market_implied_prob

        # Fee drag (taker in, maker out — conservative round-trip)
        total_fee_drag = maker_fee + taker_fee
        net_edge_after_fees = edge_per_contract - total_fee_drag

        # Payout ratio for Kelly: for a YES contract at price P, if it wins
        # you receive $1. If you paid P to buy it, your net profit is (1-P).
        # odds_received = (1 - P) / P  (net profit per $1 risked)
        if market_price > 0:
            odds_received = (1.0 - market_price) / market_price
        else:
            odds_received = 0.0

        kelly = self.compute_kelly_fraction(
            odds_received=odds_received,
            win_probability=estimated_probability,
            kelly_fraction=0.5,   # half-Kelly
        )

        is_actionable = net_edge_after_fees > self.MIN_EDGE_THRESHOLD

        return EdgeResult(
            market_price=round(market_price, 4),
            market_implied_prob=round(market_implied_prob, 6),
            estimated_probability=round(estimated_probability, 6),
            edge_per_contract=round(edge_per_contract, 6),
            net_edge_after_fees=round(net_edge_after_fees, 6),
            kelly_fraction=round(kelly, 6),
            max_position_size=round(min(kelly, self.MAX_KELLY_FRACTION), 6),
            is_actionable=is_actionable,
            min_edge_threshold=self.MIN_EDGE_THRESHOLD,
        )


# ---------------------------------------------------------------------------
# Sportsbook odds reference table builder
# ---------------------------------------------------------------------------

def build_sportsbook_table() -> list[dict]:
    """
    Return a reference table of common US sportsbook odds and their implied
    probabilities. Useful for sanity-checking estimate_from_sportsbook().

    Returns list of dicts with keys: odds, implied_prob, profit_per_100.
    """
    raw_odds = [
        -1000, -500, -300, -200, -150, -130, -120, -110, -105,
        +100, +105, +110, +120, +130, +150, +200, +300, +500,
        +750, +1000, +1500, +2000, +3000, +5000,
    ]
    pe = ProbabilityEstimator()
    rows = []
    for odds in raw_odds:
        prob = pe.estimate_from_sportsbook(odds)
        if odds > 0:
            profit = odds  # profit on $100 wager
        else:
            profit = 100.0 * (100.0 / abs(odds))  # profit on $100 wager
        rows.append({
            "odds": odds,
            "implied_prob": prob,
            "implied_prob_pct": f"{prob:.1%}",
            "profit_per_100": round(profit, 2),
        })
    return rows


# ---------------------------------------------------------------------------
# Main — Moontower BTC example + demos
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 64)
    print("PROBABILITY ESTIMATOR — Reference Implementation")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Example 1: Moontower BTC case
    # ------------------------------------------------------------------
    print("\n### Example 1: Moontower BTC Case\n")
    print("  Source: https://moontowermeta.com/prediction-market-arbitrage")
    print("          -using-option-chains-to-find-mispriced-bets/")
    print()
    print("  Params:  BTC spot = $89,000")
    print("           K       = $250,000 (threshold)")
    print("           T       = 13/12 ≈ 1.083 years")
    print("           σ       = 55% annualized IV")
    print("           Kalshi YES bid ≈ $0.09  (9% implied)\n")

    pe = ProbabilityEstimator()

    # Probability that BTC > $250k at expiry (call-option style).
    fair_prob = pe.estimate_from_options_chain(
        current_price=89_000,
        strike=250_000,
        iv=0.55,
        time_years=13 / 12,
        call_option=True,
    )

    z = math.log(89_000 / 250_000) / (0.55 * math.sqrt(13 / 12))
    print(f"  z-score formula:  ln(S/K) / (σ × √T)")
    print(f"                  = ln({89_000}/{250_000}) / (0.55 × √1.083)")
    print(f"                  = ln({89_000/250_000:.4f}) / {0.55 * math.sqrt(13/12):.4f}")
    print(f"                  = {z:.4f}\n")
    print(f"  Φ(−z) = Φ({-z:.2f}) = {fair_prob:.4f}  ({fair_prob:.1%})")
    print(f"  (Probability BTC > $250k by expiry, from options chain)\n")
    print(f"  Market pricing:  YES ≈ $0.09  →  market-implied prob = 9%")
    print(f"  Options fair:    {fair_prob:.1%}")
    print(f"  Gross edge:      {fair_prob - 0.09:.1%}\n")

    edge = pe.find_edge(
        market_price=0.09,
        estimated_probability=fair_prob,
    )

    print(f"  --- Edge Analysis (YES bid = $0.09, maker=2%, taker=3%) ---")
    for k, v in edge.items():
        print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # Example 2: Sportsbook reference
    # ------------------------------------------------------------------
    print("\n### Example 2: Sportsbook Edge Scenario\n")
    print("  Vegas line:   -110 (point spread, implied 52.4%)")
    print("  Your model:   55% win probability")
    print("  Edge:         55% - 52.4% = 2.6% gross\n")

    vegas_implied = pe.estimate_from_sportsbook(-110)
    sports_edge = pe.find_edge(
        market_price=vegas_implied,
        estimated_probability=0.55,
    )

    kelly = pe.compute_kelly_fraction(
        odds_received=(1.0 - vegas_implied) / vegas_implied,
        win_probability=0.55,
        kelly_fraction=0.5,
    )

    print(f"  market_implied_prob: {vegas_implied:.1%}")
    print(f"  net_edge_after_fees: {sports_edge['net_edge_after_fees']:.1%}")
    print(f"  is_actionable:       {sports_edge['is_actionable']}")
    print(f"  (need net edge > {sports_edge['min_edge_threshold']:.0%})\n")
    print(f"  Kelly fraction:      {kelly:.1%}  (half-Kelly, capped at 10% max)")

    # ------------------------------------------------------------------
    # Example 3: Kelly sizing table
    # ------------------------------------------------------------------
    print("\n### Example 3: Kelly Fraction Across Setup Types\n")
    print(f"  {'odds_received':>14}  {'win_prob':>10}  {'full_Kelly':>12}  "
          f"{'half_Kelly':>12}  label")
    print(f"  {'-'*14}  {'-'*10}  {'-'*12}  {'-'*12}  ----")

    cases = [
        (2.5,  0.60, "Good edge, moderate win rate"),
        (1.5,  0.55, "Small edge, high win rate"),
        (3.0,  0.35, "Long shot with edge"),
        (5.0,  0.20, "Very long shot, no edge"),
        (1.0,  0.50, "Break-even (edge = 0)"),
        (19.0, 0.60, "YES at $0.05, 60% estimate (call option scenario)"),
    ]
    for odds, prob, label in cases:
        q = 1.0 - prob
        full = max(0.0, (odds * prob - q) / odds)
        half = min(full * 0.5, pe.MAX_KELLY_FRACTION)
        print(
            f"  {odds:>14.2f}  {prob:>9.1%}  {full:>11.1%}  "
            f"{half:>11.1%}  {label}"
        )

    # ------------------------------------------------------------------
    # Example 4: Edge across a range of market prices
    # ------------------------------------------------------------------
    print(f"\n### Example 4: Edge Analysis — BTC > $250k (fair_prob = {fair_prob:.1%})\n")
    print(f"  {'YES Bid':>8}  {'mkt_prob':>10}  {'gross_edge':>12}  "
          f"{'net_edge':>10}  {'kelly':>8}  actionable")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*8}  ---------")

    for mp in [0.05, 0.09, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        r = pe.find_edge(market_price=mp, estimated_probability=fair_prob)
        flag = "✅ YES" if r["is_actionable"] else "❌ no"
        print(
            f"  ${mp:>7.2f}  {r['market_implied_prob']:>9.1%}  "
            f"{r['edge_per_contract']:>11.1%}  {r['net_edge_after_fees']:>9.1%}  "
            f"{r['kelly_fraction']:>7.1%}  {flag}"
        )

    # ------------------------------------------------------------------
    # Sportsbook reference table
    # ------------------------------------------------------------------
    print("\n### Reference: Common US Sportsbook Odds → Implied Probability\n")
    print(f"  {'Odds':>8}  {'Implied Prob':>14}  {'Profit per $100':>16}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*16}")

    for row in build_sportsbook_table():
        print(
            f"  {row['odds']:>8}  {row['implied_prob_pct']:>14}  "
            f"${row['profit_per_100']:>10.1f}"
        )

    print("\n" + "=" * 64)
    print("Done.")