"""
Polybot — Trader Intelligence & Copy Trading (Reference Architecture)
Based on: https://github.com/ent0n29/polybot
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict


class TraderProfile:
    """Tracks a trader's historical performance."""
    
    def __init__(self, wallet_address: str):
        self.wallet = wallet_address
        self.trades: List[dict] = []
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.avg_hold_time = timedelta()
    
    def add_trade(self, trade: dict):
        self.trades.append(trade)
        self._recompute_metrics()
    
    def _recompute_metrics(self):
        wins = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        self.win_rate = wins / len(self.trades) if self.trades else 0.0
        self.total_pnl = sum(t.get('pnl', 0) for t in self.trades)
    
    def sharpe_ratio(self) -> float:
        returns = [t.get('pnl', 0) for t in self.trades]
        if len(returns) < 2:
            return 0.0
        return np.mean(returns) / (np.std(returns) + 1e-8)
    
    def consistency_score(self) -> float:
        """
        Score how consistently profitable this trader is.
        Higher is better — we want steady winners, not lottery players.
        """
        if len(self.trades) < 10:
            return 0.0
        
        hourly_pnl = {}
        for t in self.trades:
            hour = t['timestamp'].hour
            hourly_pnl[hour] = hourly_pnl.get(hour, 0) + t.get('pnl', 0)
        
        positive_hours = sum(1 for v in hourly_pnl.values() if v > 0)
        return positive_hours / max(len(hourly_pnl), 1)


class TopTraderSelector:
    """Identifies top traders to mirror."""
    
    def __init__(self, min_trades: int = 50, lookback_days: int = 30):
        self.min_trades = min_trades
        self.lookback = timedelta(days=lookback_days)
    
    def score_trader(self, profile: TraderProfile) -> float:
        """
        Composite score for trader quality.
        Higher = better candidate for mirroring.
        """
        if len(profile.trades) < self.min_trades:
            return 0.0
        
        recent_trades = [t for t in profile.trades 
                        if t['timestamp'] > datetime.now() - self.lookback]
        if len(recent_trades) < self.min_trades // 2:
            return 0.0
        
        sharpe = profile.sharpe_ratio()
        consistency = profile.consistency_score()
        win_rate = profile.win_rate
        
        # Composite: weight recent performance higher
        return 0.3 * sharpe + 0.4 * consistency + 0.3 * win_rate
    
    def select_top_traders(self, profiles: List[TraderProfile],
                           n: int = 10) -> List[TraderProfile]:
        scored = [(self.score_trader(p), p) for p in profiles]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:n]]


class CopyTradeExecutor:
    """Executes mirror trades for selected top traders."""
    
    def __init__(self, scale_factor: float = 0.1):
        self.scale_factor = scale_factor  # fraction of target's position size
    
    def mirror_trade(self, target_trade: dict, our_portfolio: float) -> dict:
        """
        Generate our trade to mirror a target trader's position.
        """
        target_size = target_trade.get('size', 0)
        our_size = target_size * self.scale_factor * (our_portfolio / 1000)
        
        return {
            'market_id': target_trade['market_id'],
            'side': target_trade['side'],
            'size': our_size,
            'price': target_trade['price'],
            'type': 'copy_trade',
            'source_wallet': target_trade.get('wallet', 'unknown')
        }
