from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pandas as pd


TROY_OUNCE_GRAMS = 31.1034768
STOOQ_QUOTE_URL = "https://stooq.com/q/l/"


@dataclass(frozen=True)
class MarketQuote:
    symbol: str
    price: float
    timestamp: pd.Timestamp | None
    source: str
    currency: str
    unit: str


@dataclass(frozen=True)
class LondonGoldConversion:
    gold_usd_per_oz: float
    usd_cny: float
    cny_per_gram: float
    gold_quote_time: pd.Timestamp | None
    fx_quote_time: pd.Timestamp | None
    source: str


def convert_london_gold_to_cny_per_gram(gold_usd_per_oz: float, usd_cny: float) -> float:
    if gold_usd_per_oz <= 0:
        raise ValueError("伦敦金价格必须大于 0。")
    if usd_cny <= 0:
        raise ValueError("美元兑人民币汇率必须大于 0。")
    return gold_usd_per_oz * usd_cny / TROY_OUNCE_GRAMS


def convert_cny_per_gram_to_london_gold(cny_per_gram: float, usd_cny: float) -> float:
    if cny_per_gram <= 0:
        raise ValueError("人民币/克价格必须大于 0。")
    if usd_cny <= 0:
        raise ValueError("美元兑人民币汇率必须大于 0。")
    return cny_per_gram * TROY_OUNCE_GRAMS / usd_cny


def get_stooq_quote(symbol: str, currency: str, unit: str) -> MarketQuote:
    import requests

    url = f"{STOOQ_QUOTE_URL}?s={symbol.lower()}&f=sd2t2ohlcv&h&e=csv"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))
    if frame.empty or "Close" not in frame.columns:
        raise ValueError(f"Stooq 未返回 {symbol} 的有效行情。")

    row = frame.iloc[0]
    price = pd.to_numeric(row["Close"], errors="coerce")
    if pd.isna(price) or float(price) <= 0:
        raise ValueError(f"Stooq 返回的 {symbol} 价格无效。")

    timestamp = None
    if "Date" in frame.columns and "Time" in frame.columns:
        parsed = pd.to_datetime(f"{row['Date']} {row['Time']}", errors="coerce")
        if not pd.isna(parsed):
            timestamp = parsed

    return MarketQuote(
        symbol=str(row.get("Symbol", symbol)).upper(),
        price=float(price),
        timestamp=timestamp,
        source="Stooq CSV",
        currency=currency,
        unit=unit,
    )


def get_london_gold_conversion() -> LondonGoldConversion:
    gold = get_stooq_quote("xauusd", currency="USD", unit="troy ounce")
    fx = get_stooq_quote("usdcny", currency="CNY", unit="1 USD")
    return LondonGoldConversion(
        gold_usd_per_oz=gold.price,
        usd_cny=fx.price,
        cny_per_gram=convert_london_gold_to_cny_per_gram(gold.price, fx.price),
        gold_quote_time=gold.timestamp,
        fx_quote_time=fx.timestamp,
        source=f"{gold.source}: {gold.symbol}, {fx.symbol}",
    )
