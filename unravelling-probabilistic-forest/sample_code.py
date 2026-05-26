"""
Unravelling the Probabilistic Forest — Arb Detection (Simplified)

Integer programming approach for detecting bundle and combinatorial arbitrage
on Polymarket, based on the paper's methodology.
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass
import itertools
import pulp  # Requires: pip install pulp


@dataclass
class Condition:
    """A single condition (YES/NO pair) on Polymarket."""
    condition_id: str
    yes_best_ask: float  # Price to buy YES
    no_best_ask: float   # Price to buy NO (or compute from bid)


class ArbitrageDetector:
    """
    Detects arbitrage opportunities using integer programming.
    
    Type 1: Bundle Arbitrage — YES + NO < 1.0 within a single condition
    Type 2: Combinatorial Arbitrage — Across multiple conditions
    
    Key insight from paper: Type 1 is far more common and profitable.
    """
    
    def detect_bundle_arb(self, conditions: List[Condition]) -> List[Tuple[str, float]]:
        """Find single-condition bundle arbitrage opportunities."""
        results = []
        for c in conditions:
            total = c.yes_best_ask + c.no_best_ask
            if total < 1.0:
                profit = 1.0 - total
                results.append((c.condition_id, profit))
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def detect_combinatorial_arb_ip(self, conditions: List[Condition]) -> List[Dict]:
        """
        Detects combinatorial arbitrage using Integer Programming.
        
        Find portfolios of tokens (one from each condition) that sum to < 1.0
        with mutually exclusive outcomes.
        """
        n = len(conditions)
        
        # Create LP problem
        prob = pulp.LpProblem("CombinatorialArb", pulp.LpMinimize)
        
        # Variables: x_i = 1 if we buy YES on condition i, 0 if we buy NO
        x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)}
        
        # Cost: sum of (x_i * yes_price_i + (1-x_i) * no_price_i)
        cost_terms = []
        for i, c in enumerate(conditions):
            cost_terms.append(x[i] * c.yes_best_ask + (1 - x[i]) * c.no_best_ask)
        
        total_cost = pulp.lpSum(cost_terms)
        prob += total_cost  # Minimize cost
        
        # Constraint: cost < 1.0 - epsilon for guaranteed profit
        prob += total_cost <= 0.99
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if pulp.value(prob.objective) and pulp.value(total_cost) < 1.0:
            decision = {i: pulp.value(x[i]) for i in range(n)}
            portfolio = []
            for i, c in enumerate(conditions):
                outcome = "YES" if decision[i] == 1 else "NO"
                price = c.yes_best_ask if decision[i] == 1 else c.no_best_ask
                portfolio.append({
                    "condition_id": c.condition_id,
                    "outcome": outcome,
                    "price": price,
                })
            
            total = sum(p["price"] for p in portfolio)
            return [{
                "portfolio": portfolio,
                "total_cost": total,
                "profit": 1.0 - total,
                "edge_pct": (1.0 - total) * 100,
            }]
        
        return []


# Example usage
if __name__ == "__main__":
    detector = ArbitrageDetector()
    
    conditions = [
        Condition("btc-15m-001", yes_best_ask=0.47, no_best_ask=0.40),
        Condition("btc-15m-002", yes_best_ask=0.51, no_best_ask=0.48),
        Condition("eth-15m-001", yes_best_ask=0.44, no_best_ask=0.42),
    ]
    
    # Type 1: Bundle arb
    bundle_opportunities = detector.detect_bundle_arb(conditions)
    print("=== Bundle Arbitrage Opportunities ===")
    for cid, profit in bundle_opportunities:
        print(f"  {cid}: ${profit:.4f} profit/share")
    
    # Type 2: Combinatorial arb
    comb_opportunities = detector.detect_combinatorial_arb_ip(conditions)
    print("\n=== Combinatorial Arbitrage Opportunities ===")
    for opp in comb_opportunities:
        print(f"  Cost: ${opp['total_cost']:.2f} → Profit: ${opp['profit']:.2f}")
