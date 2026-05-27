from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "close"}


@dataclass(frozen=True)
class DataSourceInfo:
    name: str
    description: str


@dataclass(frozen=True)
class SgeDelayedQuote:
    symbol: str
    last: float
    high: float
    low: float
    open: float
    quote_date: pd.Timestamp | None
    source: str = "上海黄金交易所延时行情"


SGE_DELAYED_QUOTES_URL = "https://www.sge.com.cn/h5_sjzx/yshq"


def standardize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean daily OHLC frame with date/open/high/low/close columns."""
    if frame.empty:
        raise ValueError("数据为空，无法回测。")

    rename_map = {
        "日期": "date",
        "交易日期": "date",
        "交易时间": "date",
        "时间": "date",
        "收盘": "close",
        "收盘价": "close",
        "现价": "close",
        "开盘": "open",
        "开盘价": "open",
        "最高": "high",
        "最高价": "high",
        "最低": "low",
        "最低价": "low",
    }
    normalized = frame.rename(columns={key: value for key, value in rename_map.items() if key in frame.columns})
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    missing = REQUIRED_COLUMNS.difference(normalized.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {', '.join(sorted(missing))}")

    result = normalized.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ["open", "high", "low", "close"]:
        if column not in result.columns:
            result[column] = result["close"]
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result[["date", "open", "high", "low", "close"]]
    result = result.dropna(subset=["date", "close"])
    result = result.sort_values("date").drop_duplicates("date", keep="last")
    result = result.reset_index(drop=True)

    if len(result) < 260:
        raise ValueError("数据少于 260 个交易日，回测参考意义较弱。")
    return result


def load_csv_prices(path_or_file: str | Path | BinaryIO) -> pd.DataFrame:
    frame = pd.read_csv(path_or_file)
    return standardize_price_frame(frame)


def load_sge_prices(symbol: str = "Au99.99") -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("未安装 AKShare，请先运行 pip install -r requirements.txt。") from exc

    frame = ak.spot_hist_sge(symbol=symbol)
    return standardize_price_frame(frame)


def load_sge_delayed_quotes() -> pd.DataFrame:
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("缺少 requests 或 beautifulsoup4，无法读取上金所延时行情。") from exc

    response = requests.get(
        SGE_DELAYED_QUOTES_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    date_text = soup.select_one(".detail.yshq .date")
    quote_date = None
    if date_text is not None:
        parsed_date = pd.to_datetime(date_text.get_text(strip=True), format="%Y年%m月%d日", errors="coerce")
        if not pd.isna(parsed_date):
            quote_date = parsed_date

    rows: list[dict[str, object]] = []
    for row in soup.select("tr.ininfo"):
        cells = [cell.get_text(strip=True) for cell in row.find_all("td")]
        if len(cells) < 5:
            continue
        rows.append(
            {
                "symbol": cells[0],
                "last": pd.to_numeric(cells[1], errors="coerce"),
                "high": pd.to_numeric(cells[2], errors="coerce"),
                "low": pd.to_numeric(cells[3], errors="coerce"),
                "open": pd.to_numeric(cells[4], errors="coerce"),
                "quote_date": quote_date,
            }
        )

    quotes = pd.DataFrame(rows)
    if quotes.empty:
        raise ValueError("未能从上金所页面解析到延时行情。")
    return quotes.dropna(subset=["last"])


def get_sge_delayed_quote(symbol: str = "Au99.99") -> SgeDelayedQuote:
    quotes = load_sge_delayed_quotes()
    match = quotes.loc[quotes["symbol"] == symbol]
    if match.empty:
        available = ", ".join(quotes["symbol"].astype(str).head(8))
        raise ValueError(f"延时行情中未找到 {symbol}，可用示例: {available}")

    row = match.iloc[0]
    return SgeDelayedQuote(
        symbol=str(row["symbol"]),
        last=float(row["last"]),
        high=float(row["high"]),
        low=float(row["low"]),
        open=float(row["open"]),
        quote_date=row["quote_date"] if not pd.isna(row["quote_date"]) else None,
    )


def apply_latest_quote(prices: pd.DataFrame, quote: SgeDelayedQuote) -> pd.DataFrame:
    """Overlay the latest delayed or manually entered quote on the daily price frame."""
    if quote.last <= 0:
        return prices

    result = prices.copy()
    quote_date = quote.quote_date
    if quote_date is None:
        quote_date = pd.Timestamp.today().normalize()
    else:
        quote_date = pd.Timestamp(quote_date).normalize()

    open_price = quote.open if quote.open > 0 else quote.last
    high_price = quote.high if quote.high > 0 else max(open_price, quote.last)
    low_price = quote.low if quote.low > 0 else min(open_price, quote.last)

    if quote_date in set(result["date"].dt.normalize()):
        idx = result.index[result["date"].dt.normalize() == quote_date][-1]
        result.loc[idx, ["open", "high", "low", "close"]] = [open_price, high_price, low_price, quote.last]
    elif quote_date > result["date"].max().normalize():
        result = pd.concat(
            [
                result,
                pd.DataFrame(
                    [
                        {
                            "date": quote_date,
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": quote.last,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def generate_demo_prices(seed: int = 42, start: str = "2017-01-03") -> pd.DataFrame:
    """Generate deterministic RMB/g gold-like prices for offline demos."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=pd.Timestamp.today().normalize())
    if len(dates) < 260:
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1800)

    returns = rng.normal(0.00035, 0.009, len(dates))
    returns += 0.00025 * np.sin(np.linspace(0, 10 * np.pi, len(dates)))

    shock_windows = [
        (int(len(dates) * 0.26), int(len(dates) * 0.31), 0.0035),
        (int(len(dates) * 0.55), int(len(dates) * 0.61), -0.0025),
        (int(len(dates) * 0.76), int(len(dates) * 0.84), 0.0028),
    ]
    for start_idx, end_idx, drift in shock_windows:
        returns[start_idx:end_idx] += drift

    close = 275 * np.exp(np.cumsum(returns))
    close = pd.Series(close).rolling(2, min_periods=1).mean().to_numpy()
    open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.006, len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.006, len(dates)))

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
        }
    )


def load_prices(source: str, symbol: str = "Au99.99", csv_file: str | Path | BinaryIO | None = None) -> tuple[pd.DataFrame, DataSourceInfo]:
    if source == "akshare":
        return load_sge_prices(symbol), DataSourceInfo("AKShare / SGE", f"上海黄金交易所历史行情: {symbol}")
    if source == "csv":
        if csv_file is None:
            raise ValueError("CSV 数据源需要提供文件。")
        return load_csv_prices(csv_file), DataSourceInfo("CSV", "用户上传或本地 CSV")
    if source == "demo":
        return generate_demo_prices(), DataSourceInfo("Demo", "离线演示数据")
    raise ValueError(f"未知数据源: {source}")
