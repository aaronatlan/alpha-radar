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


# --- Crypto ---------------------------------------------------------------
#
# Identifiants CoinGecko (slugs officiels). Liste courte pour la Phase 2 :
# on couvre les têtes de gondole et deux tokens "secteurs chauds" (IA).
# Les prix / volumes seront collectés quotidiennement.


class CryptoAsset(TypedDict):
    coin_id: str         # id CoinGecko
    symbol: str          # ticker (BTC, ETH...)
    name: str
    sectors: list[str]


CRYPTO_WATCHLIST: list[CryptoAsset] = [
    {"coin_id": "bitcoin", "symbol": "BTC", "name": "Bitcoin", "sectors": []},
    {"coin_id": "ethereum", "symbol": "ETH", "name": "Ethereum", "sectors": []},
    {"coin_id": "solana", "symbol": "SOL", "name": "Solana", "sectors": []},
    {"coin_id": "render-token", "symbol": "RNDR", "name": "Render", "sectors": ["ai_ml"]},
    {"coin_id": "fetch-ai", "symbol": "FET", "name": "Fetch.ai", "sectors": ["ai_ml"]},
]


CRYPTO_COIN_IDS: list[str] = [c["coin_id"] for c in CRYPTO_WATCHLIST]


# --- Mapping ticker → CIK SEC --------------------------------------------
#
# L'API SEC EDGAR indexe les filings par CIK (Central Index Key) — pas par
# ticker. On maintient un mapping local pour la watchlist Phase 2. La
# source de vérité reste `https://www.sec.gov/files/company_tickers.json` ;
# on pourra y rapatrier dynamiquement en cas de watchlist élargie.
# CIKs doivent être stockés en string (préfixés de zéros si < 10 chiffres)
# puisque l'URL EDGAR attend 10 digits.

TICKER_TO_CIK: dict[str, str] = {
    "NVDA":  "1045810",
    "AMD":   "2488",
    "GOOGL": "1652044",
    "MSFT":  "789019",
    "META":  "1326801",
    "IBM":   "51143",
    "IONQ":  "1824920",
    "RGTI":  "1838359",
    "PLTR":  "1321655",
    "CRWD":  "1535527",
    "PANW":  "1327567",
    "ISRG":  "1035267",
    "MRNA":  "1682852",
    "CRSP":  "1674416",
    "NTLA":  "1652130",
    "RKLB":  "1819994",
    "LMT":   "936468",
    "TSLA":  "1318605",
    "AAPL":  "320193",
}


def cik_padded(cik: str) -> str:
    """SEC attend des CIK à 10 chiffres zéro-paddés."""
    return cik.zfill(10)
