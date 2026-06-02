from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .strategy import StrategyConfig, generate_signals


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def _annualized_return(equity: pd.Series, dates: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1 / 365.25)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    return float((1 + total_return) ** (1 / years) - 1)


def compute_metrics(equity_frame: pd.DataFrame, value_column: str = "equity") -> dict[str, float]:
    equity = equity_frame[value_column].astype(float)
    returns = equity.pct_change().dropna()
    volatility = float(returns.std() * math.sqrt(252)) if not returns.empty else 0.0
    sharpe = float((returns.mean() / returns.std()) * math.sqrt(252)) if returns.std() > 0 else 0.0
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "annualized_return": _annualized_return(equity, equity_frame["date"]),
        "max_drawdown": _max_drawdown(equity),
        "volatility": volatility,
        "sharpe": sharpe,
    }


def run_strategy_backtest(
    prices: pd.DataFrame,
    config: StrategyConfig,
    initial_cash: float = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    indicators = add_indicators(
        prices,
        fast_window=config.fast_window,
        slow_window=config.slow_window,
        rsi_window=config.rsi_window,
        drawdown_window=config.drawdown_window,
        volatility_window=config.volatility_window,
    )
    signals = generate_signals(indicators, config)
    data = indicators.merge(signals[["date", "signal", "target_position", "reason"]], on="date", how="left")

    cash = initial_cash
    grams = 0.0
    spread = config.spread_bps / 10_000
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    trades: list[dict[str, float | str | pd.Timestamp]] = []

    for _, row in data.iterrows():
        close = float(row["close"])
        gross_value = cash + grams * close
        current_position = 0.0 if gross_value <= 0 else (grams * close) / gross_value
        target_position = float(row["target_position"])
        trade_value = (target_position - current_position) * gross_value
        trade_grams = 0.0
        spread_cost = 0.0

        if abs(target_position - current_position) >= config.rebalance_threshold:
            if trade_value > 0 and cash > 0:
                buy_value = min(trade_value, cash)
                execution_price = close * (1 + spread)
                trade_grams = buy_value / execution_price
                cash -= buy_value
                grams += trade_grams
                spread_cost = trade_grams * (execution_price - close)
            elif trade_value < 0 and grams > 0:
                sell_value = min(abs(trade_value), grams * close)
                execution_price = close * (1 - spread)
                trade_grams = -(sell_value / close)
                grams += trade_grams
                cash += abs(trade_grams) * execution_price
                spread_cost = abs(trade_grams) * (close - execution_price)

            if abs(trade_grams) > 1e-8:
                trades.append(
                    {
                        "date": row["date"],
                        "side": "买入" if trade_grams > 0 else "卖出",
                        "price": close,
                        "grams": abs(trade_grams),
                        "signal": row["signal"],
                        "target_position": target_position,
                        "spread_cost": spread_cost,
                    }
                )

        equity = cash + grams * close
        rows.append(
            {
                "date": row["date"],
                "close": close,
                "cash": cash,
                "grams": grams,
                "equity": equity,
                "position": 0.0 if equity <= 0 else grams * close / equity,
                "target_position": target_position,
                "signal": row["signal"],
                "reason": row["reason"],
                "rsi": row["rsi"],
                "pullback": row["pullback"],
                "ma_fast": row["ma_fast"],
                "ma_slow": row["ma_slow"],
                "volatility": row["volatility"],
                "momentum_3d": row.get("momentum_3d"),
                "momentum_5d": row.get("momentum_5d"),
                "momentum_10d": row.get("momentum_10d"),
                "ma_fast_slope_5d": row.get("ma_fast_slope_5d"),
                "down_days_5": row.get("down_days_5"),
            }
        )

    equity_frame = pd.DataFrame(rows)
    trade_frame = pd.DataFrame(trades)
    metrics = compute_metrics(equity_frame)
    metrics["trades"] = float(len(trade_frame))
    metrics["final_cash"] = float(cash)
    metrics["final_grams"] = float(grams)
    return equity_frame, trade_frame, metrics


def run_buy_hold_benchmark(prices: pd.DataFrame, spread_bps: float = 35, initial_cash: float = 100_000) -> pd.DataFrame:
    first_price = float(prices.iloc[0]["close"]) * (1 + spread_bps / 10_000)
    grams = initial_cash / first_price
    frame = prices[["date", "close"]].copy()
    frame["equity"] = grams * frame["close"]
    return frame


def compare_metrics(strategy_equity: pd.DataFrame, benchmark_equity: pd.DataFrame) -> pd.DataFrame:
    strategy = compute_metrics(strategy_equity)
    benchmark = compute_metrics(benchmark_equity)
    rows = []
    for key, label in [
        ("total_return", "累计收益"),
        ("annualized_return", "年化收益"),
        ("max_drawdown", "最大回撤"),
        ("volatility", "年化波动"),
        ("sharpe", "夏普比率"),
    ]:
        rows.append({"metric": label, "strategy": strategy[key], "buy_hold": benchmark[key]})
    return pd.DataFrame(rows)
