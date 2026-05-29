"""
Copy Trading Bot — Low-Latency Trader Mirroring (Reference)
Based on: https://github.com/gamma-trade-lab/polymarket-copy-trading-bot
"""

import time
from typing import Dict, List


class TradeDetector:
    """Detects trades from target wallets in real-time."""
    
    def __init__(self, target_wallets: List[str]):
        self.targets = set(target_wallets)
        self.last_seen = {w: 0 for w in target_wallets}
    
    def detect_new_trades(self, recent_trades: List[Dict]) -> List[Dict]:
        """Filter for trades from our target wallets."""
        new_trades = []
        for trade in recent_trades:
            wallet = trade.get('maker', '')
            if wallet in self.targets:
                ts = trade.get('timestamp', 0)
                if ts > self.last_seen.get(wallet, 0):
                    self.last_seen[wallet] = ts
                    new_trades.append(trade)
        return new_trades


class ProportionalMirror:
    """Mirrors trades proportionally to our portfolio."""
    
    def __init__(self, portfolio_value: float, max_per_trade_pct: float = 0.05):
        self.portfolio = portfolio_value
        self.max_per_trade = portfolio_value * max_per_trade_pct
        self.positions: Dict[str, float] = {}  # market_id -> position
    
    def compute_mirror_order(self, target_trade: Dict) -> Dict:
        """
        Generate a proportional mirror order.
        Latency-sensitive: this must run in < 10ms.
        """
        size = target_trade.get('size', 0)
        price = target_trade.get('price', 0)
        side = target_trade.get('side', '')
        market = target_trade.get('market', '')
        
        # Scale: mirror 10-50% of target's size depending on confidence
        scaled_size = min(size * 0.25, self.max_per_trade)
        
        return {
            'market': market,
            'side': side,
            'size': scaled_size,
            'price': price,
            'type': 'market' if target_trade.get('was_market') else 'limit'
        }
