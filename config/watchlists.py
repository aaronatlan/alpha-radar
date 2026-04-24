"""Watchlist initiale d'actifs cotés.

Liste volontairement réduite à une vingtaine de tickers pour la Phase 1.
Chaque ticker est rattaché à un ou plusieurs secteurs définis dans
`config/sectors.py`. En Phase 2 cette watchlist alimentera le scoring
des actions.
"""
from __future__ import annotations

from typing import TypedDict


class WatchlistItem(TypedDict):
    ticker: str
    name: str
    sectors: list[str]
    exchange: str


STOCK_WATCHLIST: list[WatchlistItem] = [
    {"ticker": "NVDA", "name": "NVIDIA", "sectors": ["ai_ml"], "exchange": "NASDAQ"},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "sectors": ["ai_ml"], "exchange": "NASDAQ"},
    {"ticker": "GOOGL", "name": "Alphabet", "sectors": ["ai_ml", "quantum_computing"], "exchange": "NASDAQ"},
    {"ticker": "MSFT", "name": "Microsoft", "sectors": ["ai_ml", "cybersecurity"], "exchange": "NASDAQ"},
    {"ticker": "META", "name": "Meta Platforms", "sectors": ["ai_ml", "computer_vision"], "exchange": "NASDAQ"},
    {"ticker": "IBM", "name": "IBM", "sectors": ["quantum_computing", "ai_ml"], "exchange": "NYSE"},
    {"ticker": "IONQ", "name": "IonQ", "sectors": ["quantum_computing"], "exchange": "NYSE"},
    {"ticker": "RGTI", "name": "Rigetti Computing", "sectors": ["quantum_computing"], "exchange": "NASDAQ"},
    {"ticker": "PLTR", "name": "Palantir", "sectors": ["ai_ml", "cybersecurity"], "exchange": "NYSE"},
    {"ticker": "CRWD", "name": "CrowdStrike", "sectors": ["cybersecurity"], "exchange": "NASDAQ"},
    {"ticker": "PANW", "name": "Palo Alto Networks", "sectors": ["cybersecurity"], "exchange": "NASDAQ"},
    {"ticker": "ISRG", "name": "Intuitive Surgical", "sectors": ["robotics"], "exchange": "NASDAQ"},
    {"ticker": "MRNA", "name": "Moderna", "sectors": ["biotech"], "exchange": "NASDAQ"},
    {"ticker": "CRSP", "name": "CRISPR Therapeutics", "sectors": ["biotech"], "exchange": "NASDAQ"},
    {"ticker": "NTLA", "name": "Intellia Therapeutics", "sectors": ["biotech"], "exchange": "NASDAQ"},
    {"ticker": "RKLB", "name": "Rocket Lab", "sectors": ["space"], "exchange": "NASDAQ"},
    {"ticker": "LMT", "name": "Lockheed Martin", "sectors": ["space"], "exchange": "NYSE"},
    {"ticker": "TSLA", "name": "Tesla", "sectors": ["robotics", "ai_ml"], "exchange": "NASDAQ"},
    {"ticker": "AAPL", "name": "Apple", "sectors": ["ai_ml"], "exchange": "NASDAQ"},
]


WATCHLIST_TICKERS: list[str] = [item["ticker"] for item in STOCK_WATCHLIST]
