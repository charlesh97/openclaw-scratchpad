"""
Semantic Market Clustering for Prediction Markets

End-to-end pipeline: ingest markets → embed → cluster → extract dependencies → find arb.

Source: arXiv:2512.02436 — "Semantic Trading: Agentic AI for Clustering and
Relationship Discovery in Prediction Markets" (Dec 2025)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import json
import time


@dataclass
class Market:
    """A prediction market with metadata."""
    market_id: str
    question: str
    description: str
    yes_price: float
    no_price: float
    category: str = ""
    platform: str = ""  # "polymarket" or "kalshi"
    conditions: list[str] = field(default_factory=list)
    expiry: float = 0.0


@dataclass
class Dependency:
    """A logical dependency between two markets."""
    source_market: str
    target_market: str
    dep_type: str  # "implies", "contradicts", "complements", "conditional"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    direction: str = "bidirectional"  # or "source_to_target" or "target_to_source"


@dataclass
class ArbitrageSignal:
    """An arbitrage signal from constraint violation."""
    markets: list[str]
    violation_type: str
    expected_sum: float
    actual_sum: float
    expected_relationship: str
    profit_estimate: float
    confidence: float


class MarketEmbedder:
    """
    Embeds market questions into vector space for clustering.
    Uses sentence-transformers or similar in production.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed(self, market: Market) -> np.ndarray:
        """
        Embed a market's question + description into a vector.
        In production: use sentence-transformers, OpenAI embeddings, etc.
        """
        # Simulated embedding based on keyword hashing
        text = f"{market.question} {market.description}".lower()
        words = text.split()

        # Create a deterministic but meaningful embedding
        rng = np.random.RandomState(hash(text) % (2**31))
        embedding = rng.randn(self.dimension).astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def embed_batch(self, markets: list[Market]) -> np.ndarray:
        """Embed multiple markets at once."""
        return np.array([self.embed(m) for m in markets])


class MarketClusterer:
    """
    Clusters markets by semantic similarity.
    Uses simple hierarchical clustering in this prototype.
    """

    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold

    def cluster(
        self, embeddings: np.ndarray, markets: list[Market]
    ) -> dict[int, list[Market]]:
        """
        Cluster markets based on embedding similarity.

        Returns dict of cluster_id → list of markets.
        """
        n = len(markets)
        if n == 0:
            return {}

        # Compute similarity matrix
        sim_matrix = embeddings @ embeddings.T

        # Simple agglomerative clustering
        clusters = {i: [markets[i]] for i in range(n)}
        cluster_embeddings = {i: embeddings[i] for i in range(n)}

        next_id = n
        while len(clusters) > 1:
            # Find most similar pair of clusters
            best_sim = -1
            best_pair = (-1, -1)

            cluster_ids = list(clusters.keys())
            for i, ci in enumerate(cluster_ids):
                for cj in cluster_ids[i + 1:]:
                    # Average linkage similarity
                    sims = []
                    for mi in clusters[ci]:
                        for mj in clusters[cj]:
                            idx_i = markets.index(mi)
                            idx_j = markets.index(mj)
                            sims.append(sim_matrix[idx_i, idx_j])
                    avg_sim = np.mean(sims) if sims else 0

                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_pair = (ci, cj)

            if best_sim < self.similarity_threshold:
                break

            # Merge
            ci, cj = best_pair
            merged = clusters[ci] + clusters[cj]
            del clusters[ci]
            del clusters[cj]
            clusters[next_id] = merged
            next_id += 1

        return clusters


class DependencyExtractor:
    """
    Uses an LLM to extract logical dependencies between markets
    within the same cluster.

    In production, this calls an LLM API.
    """

    DEPENDENCY_PROMPT = """
You are a prediction market analyst. Analyze these related prediction markets
and identify logical dependencies between them.

MARKETS:
{market_list}

For each pair of markets, determine if there is a logical dependency:
- IMPLIES: If market A resolves YES, market B is more likely to resolve YES
- CONTRADICTS: If market A resolves YES, market B is more likely to resolve NO
- COMPLEMENTS: Markets cover aspects of the same broader event
- CONDITIONAL: Market B's probability depends on market A's resolution

Respond with JSON array of dependencies:
[
  {{
    "source": "market_id",
    "target": "market_id",
    "type": "implies|contradicts|complements|conditional",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
  }}
]
Only include dependencies with confidence > 0.5.
"""

    def extract_dependencies(
        self, cluster_markets: list[Market]
    ) -> list[Dependency]:
        """
        Extract logical dependencies from a cluster of markets.
        In production, calls an LLM API.
        """
        if len(cluster_markets) < 2:
            return []

        # Build market list for prompt
        market_list = "\n".join([
            f"- [{m.market_id}] {m.question} (YES={m.yes_price:.2f})"
            for m in cluster_markets
        ])

        prompt = self.DEPENDENCY_PROMPT.format(market_list=market_list)

        # --- SIMULATED LLM RESPONSE ---
        # In production: call LLM API here
        dependencies = self._heuristic_extraction(cluster_markets)

        return dependencies

    def _heuristic_extraction(
        self, markets: list[Market]
    ) -> list[Dependency]:
        """
        Heuristic dependency extraction (simulates LLM behavior).
        Replace with actual LLM call in production.
        """
        dependencies = []

        for i, m1 in enumerate(markets):
            for m2 in markets[i + 1:]:
                # Simple keyword overlap check
                words1 = set(m1.question.lower().split())
                words2 = set(m2.question.lower().split())
                overlap = words1 & words2

                if len(overlap) >= 3:  # Significant keyword overlap
                    # Check for contradiction patterns
                    negation_words = {"not", "no", "never", "fail", "below", "under"}
                    m1_neg = bool(words1 & negation_words)
                    m2_neg = bool(words2 & negation_words)

                    if m1_neg != m2_neg:
                        dep_type = "contradicts"
                    else:
                        dep_type = "implies"

                    confidence = min(0.8, len(overlap) / 10 + 0.3)

                    dependencies.append(Dependency(
                        source_market=m1.market_id,
                        target_market=m2.market_id,
                        dep_type=dep_type,
                        confidence=confidence,
                        reasoning=f"Shared keywords: {', '.join(list(overlap)[:5])}",
                    ))

        return dependencies


class ConstraintValidator:
    """
    Validates that market probabilities obey logical constraints
    implied by their dependencies.

    For example:
    - If A implies B, then P(A) ≤ P(B)
    - If A contradicts B, then P(A) + P(B) ≤ 1
    - For mutually exclusive conditions, sum of P = 1
    """

    def check_violations(
        self,
        dependencies: list[Dependency],
        markets_by_id: dict[str, Market],
    ) -> list[ArbitrageSignal]:
        """Check all dependencies for constraint violations."""
        violations = []

        for dep in dependencies:
            source = markets_by_id.get(dep.source_market)
            target = markets_by_id.get(dep.target_market)

            if not source or not target:
                continue

            if dep.dep_type == "implies":
                # If A implies B, then P(A) ≤ P(B)
                if source.yes_price > target.yes_price + 0.02:  # 2% threshold
                    violations.append(ArbitrageSignal(
                        markets=[dep.source_market, dep.target_market],
                        violation_type="implication_violation",
                        expected_sum=target.yes_price,
                        actual_sum=source.yes_price,
                        expected_relationship=f"P({dep.source_market}) ≤ P({dep.target_market})",
                        profit_estimate=source.yes_price - target.yes_price - 0.02,
                        confidence=dep.confidence,
                    ))

            elif dep.dep_type == "contradicts":
                # If A contradicts B, then P(A) + P(B) ≤ 1
                total = source.yes_price + target.yes_price
                if total > 1.02:  # 2% threshold
                    violations.append(ArbitrageSignal(
                        markets=[dep.source_market, dep.target_market],
                        violation_type="contradiction_violation",
                        expected_sum=1.0,
                        actual_sum=total,
                        expected_relationship=f"P(A) + P(B) ≤ 1.0",
                        profit_estimate=total - 1.0 - 0.02,
                        confidence=dep.confidence,
                    ))

        return violations


class SemanticClusterPipeline:
    """
    Complete pipeline: embed → cluster → extract dependencies → find arb.

    Usage:
        pipeline = SemanticClusterPipeline()
        pipeline.ingest(markets)
        signals = pipeline.detect_arbitrage()
    """

    def __init__(self):
        self.embedder = MarketEmbedder()
        self.clusterer = MarketClusterer(similarity_threshold=0.4)
        self.dep_extractor = DependencyExtractor()
        self.validator = ConstraintValidator()
        self.markets: list[Market] = []
        self.dependencies: list[Dependency] = []

    def ingest(self, markets: list[Market]):
        """Ingest a batch of markets."""
        self.markets = markets
        print(f"Ingested {len(markets)} markets")

    def analyze(self) -> dict:
        """Run the full analysis pipeline."""
        if not self.markets:
            return {"clusters": {}, "dependencies": [], "violations": []}

        # Step 1: Embed
        print("Embedding markets...")
        embeddings = self.embedder.embed_batch(self.markets)

        # Step 2: Cluster
        print("Clustering markets...")
        clusters = self.clusterer.cluster(embeddings, self.markets)
        print(f"Found {len(clusters)} clusters")

        # Step 3: Extract dependencies per cluster
        print("Extracting dependencies...")
        self.dependencies = []
        for cluster_id, cluster_markets in clusters.items():
            if len(cluster_markets) < 2:
                continue
            deps = self.dep_extractor.extract_dependencies(cluster_markets)
            self.dependencies.extend(deps)
            print(f"  Cluster {cluster_id}: {len(cluster_markets)} markets, "
                  f"{len(deps)} dependencies")

        # Step 4: Check constraint violations
        print("Checking constraint violations...")
        markets_by_id = {m.market_id: m for m in self.markets}
        violations = self.validator.check_violations(self.dependencies, markets_by_id)
        print(f"Found {len(violations)} violations")

        return {
            "clusters": {k: [m.market_id for m in v] for k, v in clusters.items()},
            "dependencies": self.dependencies,
            "violations": violations,
        }

    def detect_arbitrage(self) -> list[ArbitrageSignal]:
        """Run analysis and return only arbitrage signals."""
        result = self.analyze()
        return result["violations"]


# --- Demo ---
if __name__ == "__main__":
    print("=== Semantic Market Clustering Demo ===\n")

    # Create sample markets
    markets = [
        Market(
            market_id="trump_president",
            question="Will Trump win the 2028 presidential election?",
            description="Binary market on Trump winning presidency",
            yes_price=0.42, no_price=0.58, category="politics",
        ),
        Market(
            market_id="gop_senate",
            question="Will Republicans win a Senate majority in 2028?",
            description="Republican Senate control",
            yes_price=0.55, no_price=0.45, category="politics",
        ),
        Market(
            market_id="gop_house",
            question="Will Republicans hold the House in 2028?",
            description="Republican House control",
            yes_price=0.51, no_price=0.49, category="politics",
        ),
        Market(
            market_id="fed_rate_cut",
            question="Will the Fed cut rates in June 2026?",
            description="Federal Reserve rate decision",
            yes_price=0.62, no_price=0.38, category="economics",
        ),
        Market(
            market_id="sp500_above_6000",
            question="Will S&P 500 be above 6000 on July 1, 2026?",
            description="S&P 500 level",
            yes_price=0.58, no_price=0.42, category="economics",
        ),
        Market(
            market_id="btc_above_100k",
            question="Will Bitcoin be above $100k on July 1, 2026?",
            description="Bitcoin price level",
            yes_price=0.71, no_price=0.29, category="crypto",
        ),
        Market(
            market_id="trump_fired_fed_chair",
            question="Will Trump fire the Fed chair in 2026?",
            description="Fed chair dismissal",
            yes_price=0.18, no_price=0.82, category="politics",
        ),
    ]

    pipeline = SemanticClusterPipeline()
    pipeline.ingest(markets)
    result = pipeline.analyze()

    print("\n--- Clusters ---")
    for cid, members in result["clusters"].items():
        print(f"  Cluster {cid}: {members}")

    print("\n--- Dependencies ---")
    for dep in result["dependencies"]:
        print(f"  {dep.source_market} --[{dep.dep_type}]--> {dep.target_market} "
              f"(conf={dep.confidence:.2f}): {dep.reasoning}")

    print("\n--- Arbitrage Signals ---")
    for sig in result["violations"]:
        print(f"  {sig.violation_type}: {sig.markets}")
        print(f"    Expected: {sig.expected_relationship}")
        print(f"    Actual sum: {sig.actual_sum:.3f}")
        print(f"    Profit estimate: {sig.profit_estimate:.3f}")
        print(f"    Confidence: {sig.confidence:.2f}")
        print()
