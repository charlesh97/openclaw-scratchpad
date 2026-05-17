# Summary: The Anatomy of a Decentralized Prediction Market

**Authors:** Philipp Dubach et al.  
**Date:** April 2026  
**arXiv:** 2604.24366

This paper provides the first comprehensive microstructure analysis of Polymarket using 30 billion tick-level order book events and on-chain trade data across 600 markets. The 8 stylized facts documented are essential reference points for any quantitative strategy development on Polymarket.

The most actionable finding: trade direction from the public WebSocket feed (which most bots use) is only ~59% accurate. Any strategy relying on trade direction signals from the feed needs to switch to on-chain OrderFilled events.

The replication package is available at: https://github.com/philippdubach/polymarket-microstructure
