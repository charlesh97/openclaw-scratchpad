"""
Polymarket Cross-Platform Arbitrage Detector (Reference Implementation)
Based on: https://github.com/ImMike/polymarket-arbitrage
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ArbitrageOpportunity:
    buy_market: str
    sell_market: str
    buy_price: float
    sell_price: float
    edge_bps: float
    max_size: float
    platform_buy: str
    platform_sell: str
    confidence: float


class BundleArbitrageDetector:
    """Detects YES+NO price ≠ $1.00 arbitrage within a single market."""
    
    def detect_bundle_arb(self, yes_price: float, no_price: float,
                          fee_bps: float = 10.0) -> Optional[float]:
        """
        Returns risk-free profit per $1 if YES + NO < $1.00 - fees.
        """
        total = yes_price + no_price
        if total < 1.0 - fee_bps / 10000:
            return 1.0 - total - fee_bps / 10000
        return None


class CrossPlatformMatcher:
    """Matches identical markets across Polymarket and Kalshi using text similarity."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.threshold = similarity_threshold
    
    def compute_similarity(self, title_a: str, title_b: str) -> float:
        """Simple text similarity using word overlap."""
        words_a = set(title_a.lower().split())
        words_b = set(title_b.lower().split())
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for',
                      'of', 'by', 'with', 'is', 'are', 'was', 'were'}
        words_a -= stop_words
        words_b -= stop_words
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
    
    def find_matches(self, polymarket_markets: List[Dict],
                     kalshi_markets: List[Dict]) -> List[Tuple]:
        """Find matching markets across platforms."""
        matches = []
        for pm in polymarket_markets:
            for kalshi in kalshi_markets:
                sim = self.compute_similarity(pm['title'], kalshi['title'])
                if sim >= self.threshold:
                    matches.append((pm, kalshi, sim))
        return sorted(matches, key=lambda x: x[2], reverse=True)


class OrderBookScanner:
    """Scans order books for cross-platform arbitrage."""
    
    def scan_for_arb(self, polymarket_book: Dict, kalshi_book: Dict,
                     min_edge_bps: float = 50.0) -> List[ArbitrageOpportunity]:
        """
        Detect price differences between platforms for the same market.
        """
        opportunities = []
        
        pm_best_bid = float(polymarket_book.get('bids', [{}])[0].get('price', 0))
        pm_best_ask = float(polymarket_book.get('asks', [{}])[0].get('price', 1))
        ks_best_bid = float(kalshi_book.get('bids', [{}])[0].get('price', 0))
        ks_best_ask = float(kalshi_book.get('asks', [{}])[0].get('price', 1))
        
        pm_bid_size = float(polymarket_book.get('bids', [{}])[0].get('size', 0))
        pm_ask_size = float(polymarket_book.get('asks', [{}])[0].get('size', 0))
        ks_bid_size = float(kalshi_book.get('bids', [{}])[0].get('size', 0))
        ks_ask_size = float(kalshi_book.get('asks', [{}])[0].get('size', 0))
        
        # Buy Polymarket, Sell Kalshi
        if pm_best_ask < ks_best_bid:
            edge = (ks_best_bid - pm_best_ask) / pm_best_ask * 10000
            if edge >= min_edge_bps:
                size = min(pm_ask_size, ks_bid_size)
                opportunities.append(ArbitrageOpportunity(
                    buy_market="Polymarket", sell_market="Kalshi",
                    buy_price=pm_best_ask, sell_price=ks_best_bid,
                    edge_bps=edge, max_size=size,
                    platform_buy="Polymarket", platform_sell="Kalshi",
                    confidence=0.9 if edge > 100 else 0.7
                ))
        
        # Buy Kalshi, Sell Polymarket
        if ks_best_ask < pm_best_bid:
            edge = (pm_best_bid - ks_best_ask) / ks_best_ask * 10000
            if edge >= min_edge_bps:
                size = min(ks_ask_size, pm_bid_size)
                opportunities.append(ArbitrageOpportunity(
                    buy_market="Kalshi", sell_market="Polymarket",
                    buy_price=ks_best_ask, sell_price=pm_best_bid,
                    edge_bps=edge, max_size=size,
                    platform_buy="Kalshi", platform_sell="Polymarket",
                    confidence=0.9 if edge > 100 else 0.7
                ))
        
        return opportunities
