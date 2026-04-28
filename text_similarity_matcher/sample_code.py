"""
Text-Similarity Market Matching Engine
======================================
Implements Finding 3 from 2026-04-27 research.

Automatically pairs prediction markets across Polymarket and Kalshi using
text embeddings + cosine similarity. Eliminates manual URL passing.

Reference: ImMike/polymarket-arbitrage (GitHub, 2025-2026)
https://github.com/ImMike/polymarket-arbitrage

Workflow:
1. Fetch all market titles + descriptions from Polymarket API
2. Fetch all market titles + descriptions from Kalshi API
3. Encode with sentence-transformer (e.g., all-MiniLM-L6-v2)
4. Compute cosine similarity matrix
5. Threshold: flag pairs with similarity > threshold (default 0.85)
6. Filter by same underlying + temporal overlap
7. Output candidate pairs for CMRA / arbitrage checking
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

# Optional: sentence-transformers for production use
# pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class Market:
    """Minimal market descriptor for matching."""
    market_id: str
    platform: str                    # "polymarket" or "kalshi"
    title: str
    description: str = ""
    question: str = ""              # Polymarket uses 'question' field
    underlying: Optional[str] = None
    resolution_date: Optional[str] = None
    yes_price: float = 0.0
    no_price: float = 0.0
    liquidity: float = 0.0
    # Optional metadata
    tags: list[str] = field(default_factory=list)
    category: Optional[str] = None


@dataclass
class MarketPair:
    """A matched pair of markets across platforms."""
    market_a: Market
    market_b: Market
    similarity_score: float
    match_type: str          # "exact", "threshold_sibling", "semantic_cousin"
    shared_underlying: bool
    temporal_overlap: bool
    notes: str = ""


class TextSimilarityMatcher:
    """
    Match prediction markets across platforms using text embeddings.

    Architecture:
    - Uses sentence-transformers (all-MiniLM-L6-v2) if available
    - Falls back to TF-IDF + cosine similarity from scikit-learn
    - Supports simulation mode (no API calls needed for testing)
    """

    DEFAULT_THRESHOLD = 0.82        # cosine similarity threshold for candidate pairs
    SIMILARITY_ROUND = 4           # decimal places for score rounding

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = DEFAULT_THRESHOLD,
        use_simulation: bool = False,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.use_simulation = use_simulation
        self._model = None

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            if HAS_SENTENCE_TRANSFORMERS:
                print(f"Loading sentence-transformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            else:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers "
                    "Or use TF-IDF fallback mode."
                )
        return self._model

    def _text_for_market(self, market: Market) -> str:
        """
        Combine title + description + question into a single text blob.
        Polymarket uses 'question' as the primary trading question.
        """
        parts = [market.title]
        if market.question and market.question != market.title:
            parts.append(market.question)
        if market.description:
            parts.append(market.description)
        if market.tags:
            parts.append(" ".join(market.tags))
        return " | ".join(parts).strip()

    def embed_markets(self, markets: list[Market]) -> dict[str, list[float]]:
        """Embed all markets and return market_id -> embedding dict."""
        texts = [self._text_for_market(m) for m in markets]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return {
            m.market_id: emb.tolist()
            for m, emb in zip(markets, embeddings)
        }

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return round(dot / (norm_a * norm_b), TextSimilarityMatcher.SIMILARITY_ROUND)

    def find_pairs(
        self,
        polymarket_markets: list[Market],
        kalshi_markets: list[Market],
        embedding_cache: Optional[dict[str, list[float]]] = None,
    ) -> list[MarketPair]:
        """
        Find all cross-platform market pairs above similarity threshold.

        Args:
            polymarket_markets: List of Polymarket markets
            kalshi_markets: List of Kalshi markets
            embedding_cache: Optional pre-computed embeddings to reuse

        Returns:
            List of MarketPair objects sorted by similarity descending
        """
        all_markets = polymarket_markets + kalshi_markets

        # Compute or load embeddings
        if embedding_cache is None:
            print("Computing embeddings (this may take a minute)...")
            embedding_cache = self.embed_markets(all_markets)

        pairs: list[MarketPair] = []

        for poly_m in polymarket_markets:
            poly_emb = embedding_cache.get(poly_m.market_id)
            if poly_emb is None:
                continue

            best_match: Optional[MarketPair] = None
            best_score = 0.0

            for kalshi_m in kalshi_markets:
                kalshi_emb = embedding_cache.get(kalshi_m.market_id)
                if kalshi_emb is None:
                    continue

                score = self.cosine_similarity(poly_emb, kalshi_emb)

                if score >= self.threshold:
                    # Determine match type
                    same_underlying = (
                        poly_m.underlying is not None
                        and kalshi_m.underlying is not None
                        and poly_m.underlying.lower() == kalshi_m.underlying.lower()
                    )

                    temporal_overlap = (
                        poly_m.resolution_date is not None
                        and kalshi_m.resolution_date is not None
                        and poly_m.resolution_date == kalshi_m.resolution_date
                    )

                    # Determine match type
                    if same_underlying and temporal_overlap:
                        match_type = "exact"
                    elif same_underlying:
                        match_type = "threshold_sibling"
                    else:
                        match_type = "semantic_cousin"

                    pair = MarketPair(
                        market_a=poly_m,
                        market_b=kalshi_m,
                        similarity_score=score,
                        match_type=match_type,
                        shared_underlying=same_underlying,
                        temporal_overlap=temporal_overlap,
                        notes=f"Score={score:.4f}, type={match_type}",
                    )
                    pairs.append(pair)

                    if score > best_score:
                        best_score = score
                        best_match = pair

        # Sort by similarity descending
        pairs.sort(key=lambda p: p.similarity_score, reverse=True)
        return pairs

    def filter_arbitrage_candidates(
        self,
        pairs: list[MarketPair],
        price_deviation_threshold: float = 0.02,
    ) -> list[MarketPair]:
        """
        Filter matched pairs for likely arbitrage opportunities.
        Only include pairs where price deviation exceeds threshold.
        """
        candidates = []
        for pair in pairs:
            if not (pair.shared_underlying and pair.temporal_overlap):
                continue

            m_a = pair.market_a
            m_b = pair.market_b

            price_diff = abs(m_a.yes_price - m_b.yes_price)
            if price_diff >= price_deviation_threshold:
                pair.notes += f" | price_diff={price_diff:.4f} > {price_deviation_threshold:.4f}"
                candidates.append(pair)

        return candidates


# -------------------------------------------------------------------
# TF-IDF fallback when sentence-transformers is not available
# -------------------------------------------------------------------

class TfidfMatcher:
    """
    TF-IDF based matcher as fallback when sentence-transformers unavailable.
    Less accurate but requires only scikit-learn (pip install scikit-learn).
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray().tolist()

    def find_pairs_tfidf(
        self,
        polymarket_markets: list[Market],
        kalshi_markets: list[Market],
    ) -> list[MarketPair]:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        all_texts = [self._text(m) for m in polymarket_markets + kalshi_markets]
        vectors = self.fit_transform(all_texts)

        poly_vectors = vectors[:len(polymarket_markets)]
        kalshi_vectors = vectors[len(polymarket_markets):]

        pairs = []
        for i, poly_m in enumerate(polymarket_markets):
            for j, kalshi_m in enumerate(kalshi_markets):
                score = cosine_similarity([poly_vectors[i]], [kalshi_vectors[j]])[0][0]
                if score >= self.threshold:
                    pairs.append(MarketPair(
                        market_a=poly_m,
                        market_b=kalshi_m,
                        similarity_score=round(float(score), 4),
                        match_type="tfidf_cosine",
                        shared_underlying=False,
                        temporal_overlap=False,
                    ))
        pairs.sort(key=lambda p: p.similarity_score, reverse=True)
        return pairs

    @staticmethod
    def _text(market: Market) -> str:
        parts = [market.title]
        if market.question:
            parts.append(market.question)
        if market.description:
            parts.append(market.description)
        return " ".join(parts)


# -------------------------------------------------------------------
# Demo with synthetic data
# -------------------------------------------------------------------

def demo():
    """Demonstrate market matching with synthetic data."""
    kalshi_markets = [
        Market(
            market_id="kalshi-btc-100k-dec31",
            platform="kalshi",
            title="Will Bitcoin be above $100,000 on December 31, 2026?",
            underlying="BTC",
            resolution_date="2026-12-31",
            yes_price=0.84,
            liquidity=100_000,
            tags=["crypto", "bitcoin", "price"],
        ),
        Market(
            market_id="kalshi-btc-110k-dec31",
            platform="kalshi",
            title="Will Bitcoin be above $110,000 on December 31, 2026?",
            underlying="BTC",
            resolution_date="2026-12-31",
            yes_price=0.72,
            liquidity=80_000,
            tags=["crypto", "bitcoin", "price"],
        ),
        Market(
            market_id="kalshi-fed-rate-dec26",
            platform="kalshi",
            title="Will the Fed cut rates by December 2026?",
            underlying="FED",
            resolution_date="2026-12-31",
            yes_price=0.55,
            liquidity=50_000,
            tags=["macro", "fed", "rates"],
        ),
    ]

    polymarket_markets = [
        Market(
            market_id="poly-btc-100k-dec31",
            platform="polymarket",
            title="Will BTC close above $100,000 by Dec 31 2026?",
            question="Will BTC close above $100,000 by Dec 31 2026?",
            underlying="BTC",
            resolution_date="2026-12-31",
            yes_price=0.83,   # slight difference from Kalshi
            liquidity=120_000,
        ),
        Market(
            market_id="poly-btc-110k-dec31",
            platform="polymarket",
            title="Will BTC close above $110,000 by Dec 31 2026?",
            question="Will BTC close above $110,000 by Dec 31 2026?",
            underlying="BTC",
            resolution_date="2026-12-31",
            yes_price=0.75,
            liquidity=90_000,
        ),
        Market(
            market_id="poly-fed-cut-2026",
            platform="polymarket",
            title="Fed Rate Cut by December 2026?",
            question="Will the Federal Reserve cut interest rates in 2026?",
            underlying="FED",
            resolution_date="2026-12-31",
            yes_price=0.52,
            liquidity=60_000,
        ),
        Market(
            market_id="poly-btc-summer-2026",
            platform="polymarket",
            title="Will BTC reach $200,000 by Summer 2026?",
            underlying="BTC",
            resolution_date="2026-06-01",
            yes_price=0.25,
            liquidity=30_000,
        ),
    ]

    print(f"\n=== Text Similarity Market Matcher Demo ===")
    print(f"Polymarket markets: {len(polymarket_markets)}")
    print(f"Kalshi markets: {len(kalshi_markets)}")

    if HAS_SENTENCE_TRANSFORMERS:
        print("\nUsing sentence-transformer (all-MiniLM-L6-v2)")
        matcher = TextSimilarityMatcher(threshold=0.80)
        pairs = matcher.find_pairs(polymarket_markets, kalshi_markets)
        candidates = matcher.filter_arbitrage_candidates(pairs, price_deviation_threshold=0.015)
    else:
        print("\nsentence-transformers not available — using TF-IDF fallback")
        matcher = TfidfMatcher(threshold=0.70)
        pairs = matcher.find_pairs_tfidf(polymarket_markets, kalshi_markets)
        candidates = []

    print(f"\nAll matched pairs (similarity >= {matcher.threshold}): {len(pairs)}")
    for pair in pairs:
        print(f"\n  [{pair.match_type}] {pair.market_a.market_id} ↔ {pair.market_b.market_id}")
        print(f"    Similarity: {pair.similarity_score:.4f}")
        print(f"    Shared underlying: {pair.shared_underlying}")
        print(f"    Temporal overlap: {pair.temporal_overlap}")
        print(f"    Price diff: {abs(pair.market_a.yes_price - pair.market_b.yes_price):.4f}")

    if candidates:
        print(f"\nArb candidates (price diff > 1.5%): {len(candidates)}")
        for c in candidates:
            print(f"  {c.market_a.market_id} vs {c.market_b.market_id}: price_diff={c.notes.split('price_diff=')[-1]}")


if __name__ == "__main__":
    demo()
