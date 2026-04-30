"""
LLM Dependency Graph for Combinatorial Arbitrage — Reference Implementation
============================================================================
Builds a semantic dependency graph of prediction markets to reduce the
combinatorial search space from O(2^(n+m)) to O(n × k) where k << n.

Architecture:
  1. Fetch market corpus (Polymarket + Kalshi)
  2. Embed all question texts with sentence-transformers
  3. Similarity pre-filter: only pairs above threshold advance
  4. LLM Dependency Classifier: determine relationship type + confidence
  5. Graph construction: store edges with metadata
  6. CMRA scanner: run targeted arbitrage check only on connected pairs

Author: vega research
Source: arXiv:2508.03474, arXiv:2602.07048
Requirements: sentence-transformers, openai (or anthropic), networkx
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class DependencyType(str, Enum):
    IMPLIES = "IMPLIES"        # A resolution implies B (if A=YES → B=YES)
    COMPLEMENT = "COMPLEMENT"  # Markets address different aspects of same topic
    CONTRADICTION = "CONTRADICTION"  # Markets are logical inverses
    NONE = "NONE"              # No dependency
    UNCERTAIN = "UNCERTAIN"     # LLM couldn't determine


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Market:
    condition_id: str
    question: str
    platform: str          # "polymarket" | "kalshi"
    end_date: str
    volume: float
    embedding: Optional[list[float]] = None

    @property
    def text_key(self) -> str:
        return self.question.strip()


@dataclass
class DependencyEdge:
    """A directed edge in the market dependency graph."""
    source_id: str
    target_id: str
    dep_type: DependencyType
    confidence: Confidence
    reasoning: str
    similarity_score: float = 0.0
    detected_at: float = field(default_factory=time.time)
    ttl_days: int = 7  # edges expire after 7 days by default

    def is_expired(self) -> bool:
        return (time.time() - self.detected_at) > (self.ttl_days * 86400)

    def is_actionable(self, min_confidence: Confidence = Confidence.MEDIUM) -> bool:
        order = {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1}
        return order[self.confidence] >= order[min_confidence]


# ---------------------------------------------------------------------------
# Layer 1: Similarity Pre-filter
# ---------------------------------------------------------------------------

class SimilarityFilter:
    """
    Reduces O(n²) market pair comparisons using embedding similarity.
    Uses sentence-transformers for fast, high-quality text embeddings.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.78,
        max_neighbors: int = 50,
    ):
        self.threshold = similarity_threshold
        self.max_neighbors = max_neighbors
        self.model_name = model_name
        self.model = None  # loaded lazily
        self.logger = logging.getLogger("SimilarityFilter")

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Loaded embedding model: {self.model_name}")

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of market questions into embedding vectors."""
        self._load_model()
        import numpy as np
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def find_candidate_pairs(
        self, markets: list[Market]
    ) -> list[tuple[str, str, float]]:
        """
        Returns list of (market_id_a, market_id_b, similarity_score) pairs
        that pass the similarity threshold.
        """
        self.logger.info(f"Encoding {len(markets)} markets...")
        texts = [m.text_key for m in markets]
        embeddings = await self.encode_batch(texts)

        # Attach embeddings back to markets
        for m, emb in zip(markets, embeddings):
            m.embedding = emb

        # Find candidate pairs above threshold
        candidates = []
        n = len(markets)

        # Chunk processing to avoid O(n²) explosion
        for i in range(0, n, 100):
            chunk = markets[i : i + 100]
            for j, m_b in enumerate(markets):
                if j < i:  # avoid duplicates
                    continue
                for k, m_a in enumerate(chunk):
                    if m_a.condition_id == m_b.condition_id:
                        continue
                    score = self.cosine_similarity(m_a.embedding, m_b.embedding)
                    if score >= self.threshold:
                        # Keep only top-k per market to limit LLM calls
                        candidates.append((m_a.condition_id, m_b.condition_id, score))

        # Sort by score, limit per-market neighbors
        candidates.sort(key=lambda x: x[2], reverse=True)
        limited = self._limit_per_market(candidates)

        self.logger.info(
            f"Pre-filter reduced {n*(n-1)//2} pairs → {len(limited)} candidates "
            f"(threshold={self.threshold})"
        )
        return limited

    def _limit_per_market(
        self, pairs: list[tuple[str, str, float]]
    ) -> list[tuple[str, str, float]]:
        """Keep only top-k neighbors per market to bound LLM calls."""
        from collections import defaultdict
        neighbors: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        for a_id, b_id, score in pairs:
            neighbors[a_id].append((a_id, b_id, score))
            neighbors[b_id].append((a_id, b_id, score))

        selected = []
        for m_id, neighs in neighbors.items():
            top_k = sorted(neighs, key=lambda x: x[2], reverse=True)[: self.max_neighbors]
            selected.extend(top_k)

        # Deduplicate
        seen = set()
        unique = []
        for p in selected:
            key = tuple(sorted([p[0], p[1]]))
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique


# ---------------------------------------------------------------------------
# Layer 2: LLM Dependency Classifier
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a prediction market dependency analyst. Your job is to determine whether the resolution of one market logically implies or relates to the outcome of another.

Analyze the two market questions and output a structured classification.
Be precise and conservative — only classify HIGH confidence dependencies.
Relationships should be grounded in objective logic, not correlation or coincidence.

Output format:
DEP_TYPE: <IMPLIES|COMPLEMENT|CONTRADICTION|NONE>
CONFIDENCE: <HIGH|MEDIUM|LOW>
REASONING: <2 sentences maximum>"""


USER_TEMPLATE = """Market A: "{q_a}"
Market B: "{q_b}"

Does Market A's resolution logically imply or relate to Market B's outcome? How?"""


LLM_CLASSIFICATION_PROMPT = """Given these two prediction markets:

Market A: "{q_a}"
Market B: "{q_b}"

Classify their relationship.

Rules:
- IMPLIES: If YES on Market A, Market B must also resolve YES (or makes it far more likely)
- CONTRADICTION: Markets cannot both be YES simultaneously
- COMPLEMENT: Markets address related aspects of the same underlying topic but don't strictly imply each other
- NONE: No logical dependency relationship

Be conservative. Only HIGH confidence if the logical relationship is clear and unambiguous.
"""


class LLMDependencyClassifier:
    """
    Uses an LLM to classify market dependency relationships.
    Falls back to keyword heuristics if LLM is unavailable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        batch_size: int = 20,
        rate_limit_rpm: int = 60,
    ):
        self.api_key = api_key or self._load_api_key()
        self.model = model
        self.batch_size = batch_size
        self.rate_limit_rpm = rate_limit_rpm
        self.logger = logging.getLogger("LLMClassifier")
        self._call_count = 0

    def _load_api_key(self) -> Optional[str]:
        import os
        return os.environ.get("OPENAI_API_KEY")

    async def classify_pair(
        self, market_a: Market, market_b: Market
    ) -> DependencyEdge:
        """Classify a single market pair. Returns a DependencyEdge."""
        self._call_count += 1

        if not self.api_key:
            return self._heuristic_fallback(market_a, market_b)

        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": LLM_CLASSIFICATION_PROMPT.format(
                            q_a=market_a.question, q_b=market_b.question
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=200,
            )
            text = response.choices[0].message.content or ""

            return self._parse_llm_response(
                text, market_a.condition_id, market_b.condition_id, similarity_score=0.0
            )

        except Exception as e:
            self.logger.warning(f"LLM call failed: {e}")
            return self._heuristic_fallback(market_a, market_b)

    def _parse_llm_response(
        self, text: str, source_id: str, target_id: str, similarity_score: float
    ) -> DependencyEdge:
        """Parse LLM raw text output into a DependencyEdge."""
        dep_type = DependencyType.UNCERTAIN
        confidence = Confidence.LOW
        reasoning = "Parse failed"

        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("DEP_TYPE:") or line.startswith("Type:"):
                val = line.split(":", 1)[1].strip().upper()
                if val in [e.value for e in DependencyType]:
                    dep_type = DependencyType(val)
            elif line.startswith("CONFIDENCE:") or line.startswith("Confidence:"):
                val = line.split(":", 1)[1].strip().upper()
                if val in [e.value for e in Confidence]:
                    confidence = Confidence(val)
            elif line.startswith("REASONING:") or line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

        return DependencyEdge(
            source_id=source_id,
            target_id=target_id,
            dep_type=dep_type,
            confidence=confidence,
            reasoning=reasoning,
            similarity_score=similarity_score,
        )

    def _heuristic_fallback(
        self, market_a: Market, market_b: Market
    ) -> DependencyEdge:
        """
        Keyword + structure based fallback when LLM is unavailable.
        Less accurate but functional for testing.
        """
        qa = market_a.question.lower()
        qb = market_b.question.lower()

        # Check for threshold nesting
        # "BTC > $100k" implies "BTC > $90k" if same date
        try:
            import re

            def extract_threshold(q):
                m = re.search(r"(?:>|above|over)\s*\$?([\d,.]+[km]?)", q)
                return float(m.group(1).replace(",", "").replace("k", "000").replace("m", "000000")) if m else None

            ta = extract_threshold(qa)
            tb = extract_threshold(qb)
            if ta and tb and abs(ta - tb) < ta * 0.1:
                if ta > tb:
                    return DependencyEdge(
                        source_id=market_a.condition_id,
                        target_id=market_b.condition_id,
                        dep_type=DependencyType.IMPLIES,
                        confidence=Confidence.MEDIUM,
                        reasoning=f"Threshold {ta} implies threshold {tb}",
                    )
                elif tb > ta:
                    return DependencyEdge(
                        source_id=market_b.condition_id,
                        target_id=market_a.condition_id,
                        dep_type=DependencyType.IMPLIES,
                        confidence=Confidence.MEDIUM,
                        reasoning=f"Threshold {tb} implies threshold {ta}",
                    )
        except Exception:
            pass

        # Default: no dependency detected without LLM
        return DependencyEdge(
            source_id=market_a.condition_id,
            target_id=market_b.condition_id,
            dep_type=DependencyType.NONE,
            confidence=Confidence.LOW,
            reasoning="Heuristic fallback — no LLM",
        )

    async def classify_batch(
        self,
        market_pairs: list[tuple[Market, Market, float]],
    ) -> list[DependencyEdge]:
        """Process a batch of market pairs, rate-limited."""
        results = []
        for i in range(0, len(market_pairs), self.batch_size):
            batch = market_pairs[i : i + self.batch_size]
            tasks = [
                self.classify_pair(m_a, m_b)
                for (m_a, m_b, sim) in batch
            ]
            edges = await asyncio.gather(*tasks)
            for edge, (_, _, sim) in zip(edges, batch):
                edge.similarity_score = sim
                results.append(edge)

            self.logger.info(
                f"Batch {i//self.batch_size + 1}: classified {len(batch)} pairs"
            )
            await asyncio.sleep(60.0 / self.rate_limit_rpm)  # rate limit

        return results


# ---------------------------------------------------------------------------
# Layer 3: Dependency Graph Construction
# ---------------------------------------------------------------------------

class MarketDependencyGraph:
    """
    Directed graph of market dependencies.
    Built on top of NetworkX for efficient traversal and analysis.
    """

    def __init__(self):
        import networkx as nx
        self.G = nx.DiGraph()
        self.logger = logging.getLogger("DepGraph")
        self._market_lookup: dict[str, Market] = {}

    def add_market(self, market: Market):
        self._market_lookup[market.condition_id] = market
        self.G.add_node(market.condition_id, question=market.question, platform=market.platform)

    def add_edge(self, edge: DependencyEdge):
        if not edge.is_actionable():
            return
        if edge.dep_type == DependencyType.NONE:
            return
        self.G.add_edge(
            edge.source_id,
            edge.target_id,
            dep_type=edge.dep_type.value,
            confidence=edge.confidence.value,
            reasoning=edge.reasoning,
            similarity=edge.similarity_score,
        )

    def get_neighbors(
        self, condition_id: str, min_confidence: Confidence = Confidence.MEDIUM
    ) -> list[tuple[str, DependencyType]]:
        """Get all markets that depend on (or are depended upon by) this market."""
        neighbors = []
        for _, tgt, data in self.G.out_edges(condition_id, data=True):
            conf = Confidence(data.get("confidence", "LOW"))
            if conf.value >= min_confidence.value:
                neighbors.append((tgt, DependencyType(data.get("dep_type", "NONE"))))
        return neighbors

    def get_high_confidence_paths(self, max_hops: int = 2) -> list[list[str]]:
        """Get all HIGH confidence dependency chains of up to max_hops."""
        paths = []
        for source in self.G.nodes:
            for target in self.G.nodes:
                if source == target:
                    continue
                try:
                    for path in nx.all_simple_paths(self.G, source, target, cutoff=max_hops):
                        if all(
                            self.G.edges[p[i], p[i + 1]].get("confidence") == "HIGH"
                            for i in range(len(p) - 1)
                        ):
                            paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        return paths

    def stats(self) -> dict:
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "high_conf_edges": sum(
                1 for _, _, d in self.G.edges(data=True) if d.get("confidence") == "HIGH"
            ),
            "implication_edges": sum(
                1 for _, _, d in self.G.edges(data=True) if d.get("dep_type") == "IMPLIES"
            ),
        }

    def export_json(self, filepath: str):
        import networkx as nx
        data = nx.node_link_data(self.G)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Exported graph to {filepath}")


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

async def build_dependency_graph(
    markets: list[Market],
    similarity_threshold: float = 0.78,
    max_neighbors: int = 50,
    openai_api_key: Optional[str] = None,
) -> MarketDependencyGraph:
    """
    Full pipeline:
    1. Similarity pre-filter → candidate pairs
    2. LLM dependency classification
    3. Graph construction
    """
    logger = logging.getLogger("Pipeline")
    logger.info(f"Starting dependency graph build for {len(markets)} markets")

    graph = MarketDependencyGraph()
    for m in markets:
        graph.add_market(m)

    # Step 1: Similarity pre-filter
    filter_ = SimilarityFilter(
        similarity_threshold=similarity_threshold,
        max_neighbors=max_neighbors,
    )
    candidate_pairs = await filter_.find_candidate_pairs(markets)
    logger.info(f"Candidate pairs for LLM classification: {len(candidate_pairs)}")

    if not candidate_pairs:
        logger.warning("No candidate pairs found. Try lowering similarity threshold.")
        return graph

    # Build market lookup
    market_lookup = {m.condition_id: m for m in markets}

    # Step 2: LLM classification
    pairs_to_classify = []
    for a_id, b_id, score in candidate_pairs:
        if a_id in market_lookup and b_id in market_lookup:
            pairs_to_classify.append(
                (market_lookup[a_id], market_lookup[b_id], score)
            )

    classifier = LLMDependencyClassifier(api_key=openai_api_key)
    edges = await classifier.classify_batch(pairs_to_classify)

    # Step 3: Build graph
    for edge in edges:
        graph.add_edge(edge)

    logger.info(f"Graph built: {graph.stats()}")
    return graph


# ---------------------------------------------------------------------------
# CMRA integration hook
# ---------------------------------------------------------------------------

def check_cmra_opportunity_on_edge(
    graph: MarketDependencyGraph,
    source_id: str,
    target_id: str,
    source_price: float,
    target_price: float,
) -> Optional[dict]:
    """
    Given a dependency edge (A → B), check if CMRA opportunity exists.

    If A implies B, then P(B) should be >= P(A).
    If P(B) < P(A), there's a combinatorial arb: buy B, sell A (or vice versa).

    This is a stub — wire into actual CMRA detector for real prices.
    """
    edge_data = graph.G.edges[source_id, target_id]
    dep_type = DependencyType(edge_data.get("dep_type", "NONE"))

    if dep_type == DependencyType.IMPLIES:
        if target_price < source_price:
            # B is cheaper than A even though A implies B → arb exists
            gap = source_price - target_price
            return {
                "type": "combinatorial",
                "action": "BUY target / SELL source",
                "source_id": source_id,
                "target_id": target_id,
                "gap_pct": gap * 100,
                "edge_confidence": edge_data.get("confidence"),
            }
    return None


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

async def demo():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("Demo")

    # Synthetic market corpus for demonstration
    demo_markets = [
        Market("c1", "Will BTC close above $100,000 by December 31, 2025?", "polymarket", "2025-12-31", 1_000_000),
        Market("c2", "Will BTC close above $95,000 by November 30, 2025?", "polymarket", "2025-11-30", 800_000),
        Market("c3", "Will BTC close above $90,000 by October 31, 2025?", "polymarket", "2025-10-31", 600_000),
        Market("c4", "Will the Fed hold rates at the September 2025 meeting?", "kalshi", "2025-09-01", 2_000_000),
        Market("c5", "Will the Fed cut rates by 25bp at the September 2025 meeting?", "kalshi", "2025-09-01", 1_500_000),
        Market("c6", "Will BTC close above $80,000 by September 1, 2025?", "polymarket", "2025-09-01", 400_000),
        Market("c7", "Will ETH surpass $4,000 by end of 2025?", "polymarket", "2025-12-31", 500_000),
        Market("c8", "Will the S&P 500 close above 6000 by December 31, 2025?", "polymarket", "2025-12-31", 3_000_000),
    ]

    logger.info(f"Running pipeline on {len(demo_markets)} demo markets...")

    graph = await build_dependency_graph(
        demo_markets,
        similarity_threshold=0.60,  # Lower for demo with small corpus
        max_neighbors=10,
    )

    print(f"\n{'='*60}")
    print(f"GRAPH STATS: {graph.stats()}")
    print(f"\nEdges in graph:")
    for u, v, d in graph.G.edges(data=True):
        print(f"  {u} → {v} [{d.get('dep_type')}, {d.get('confidence')}]")
        print(f"    Reasoning: {d.get('reasoning', 'N/A')}")

    if graph.G.number_of_edges() == 0:
        print("\nNote: No edges found — this is expected for the small demo corpus")
        print("      Run with larger market corpus and OpenAI API key for full results.")


if __name__ == "__main__":
    try:
        asyncio.run(demo())
    except Exception as e:
        print(f"Demo error: {e}")
