from __future__ import annotations

import numpy as np
import pandas as pd


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_indicators(
    prices: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 120,
    rsi_window: int = 14,
    drawdown_window: int = 252,
    volatility_window: int = 20,
) -> pd.DataFrame:
    frame = prices.copy()
    frame["return"] = frame["close"].pct_change().fillna(0)
    frame["ma_fast"] = frame["close"].rolling(fast_window, min_periods=fast_window).mean()
    frame["ma_slow"] = frame["close"].rolling(slow_window, min_periods=slow_window).mean()
    frame["rsi"] = relative_strength_index(frame["close"], rsi_window)
    frame["rolling_high"] = frame["close"].rolling(drawdown_window, min_periods=20).max()
    frame["pullback"] = frame["close"] / frame["rolling_high"] - 1
    frame["volatility"] = frame["return"].rolling(volatility_window, min_periods=volatility_window).std() * np.sqrt(252)
    frame["distance_to_fast_ma"] = frame["close"] / frame["ma_fast"] - 1
    frame["distance_to_slow_ma"] = frame["close"] / frame["ma_slow"] - 1
    return frame
