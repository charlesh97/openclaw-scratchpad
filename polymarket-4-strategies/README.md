# Polymarket 4-Strategy Bot (MrFadiAi)

**Source:** https://github.com/MrFadiAi/Polymarket-bot

## What It Does
The "ultimate" open-source automated Polymarket bot with 4 built-in trading strategies and comprehensive risk management (4-layer protection system). Node.js-based, built for easy setup.

## 4 Strategies
1. **Copy Trading:** Follow top Polymarket traders with smart filtering (>60% win rate, 1.5x profit factor)
2. **Market Making:** Liquidity provision with bid-ask spread capture
3. **Dip Arbitrage (DipArb):** Buy dips in short-term markets
4. **Information Arbitrage:** Trade on news/event data before the market reacts

## Risk Management (4-Layer)
- Daily loss limit: 5%
- Monthly loss limit: 15%
- Maximum drawdown: 25%
- Total loss halt: 40%

## Implementability: 4/5
**MEDIUM** — Node.js, easy setup (git clone + npm install + config). The risk management system is production-ready. Smart Money filtering is a highlight (avoids lucky whale following). The 4 strategies cover most common approaches.

## Next Steps
1. Review the Smart Money filtering logic — this is the key IP
2. Test each strategy in a simulated environment
3. Customize the risk parameters for your portfolio size
