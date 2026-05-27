from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Iterable

import pandas as pd

from .backtest import run_strategy_backtest
from .indicators import add_indicators
from .strategy import StrategyConfig, generate_signals


DEFAULT_FORWARD_HORIZONS = (1, 5, 20, 60)


def evaluate_forward_returns(
    prices: pd.DataFrame,
    config: StrategyConfig,
    horizons: Iterable[int] = DEFAULT_FORWARD_HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate how each generated signal performed after fixed future windows."""
    horizon_days = tuple(int(horizon) for horizon in horizons)
    indicators = add_indicators(
        prices,
        fast_window=config.fast_window,
        slow_window=config.slow_window,
        rsi_window=config.rsi_window,
        drawdown_window=config.drawdown_window,
        volatility_window=config.volatility_window,
    )
    signals = generate_signals(indicators, config)
    data = indicators.merge(
        signals[["date", "signal", "target_position", "reason"]],
        on="date",
        how="left",
    )

    for horizon in horizon_days:
        data[f"return_{horizon}d"] = data["close"].shift(-horizon) / data["close"] - 1

    summary_rows: list[dict[str, float | int | str]] = []
    for signal, group in data.groupby("signal", sort=False):
        for horizon in horizon_days:
            returns = group[f"return_{horizon}d"].dropna().astype(float)
            if returns.empty:
                continue
            summary_rows.append(
                {
                    "signal": signal,
                    "horizon_days": horizon,
                    "samples": int(len(returns)),
                    "avg_return": float(returns.mean()),
                    "median_return": float(returns.median()),
                    "win_rate": float((returns > 0).mean()),
                    "best_return": float(returns.max()),
                    "worst_return": float(returns.min()),
                }
            )

    return data, pd.DataFrame(summary_rows)


def run_parameter_grid(
    prices: pd.DataFrame,
    base_config: StrategyConfig,
    initial_cash: float,
    fast_windows: Iterable[int] = (10, 20, 30, 40),
    slow_windows: Iterable[int] = (80, 120, 160, 200),
    pullback_pcts: Iterable[float] = (0.05, 0.08, 0.12),
) -> pd.DataFrame:
    """Run a compact grid search for strategy parameters."""
    rows: list[dict[str, float | int]] = []
    for fast_window, slow_window, pullback_pct in product(fast_windows, slow_windows, pullback_pcts):
        if fast_window >= slow_window:
            continue

        config = replace(
            base_config,
            fast_window=int(fast_window),
            slow_window=int(slow_window),
            deep_pullback_pct=float(pullback_pct),
        )
        equity, trades, metrics = run_strategy_backtest(prices, config, initial_cash=initial_cash)
        latest = equity.iloc[-1]
        score = (
            metrics["annualized_return"]
            + metrics["max_drawdown"] * 0.20
            + metrics["sharpe"] * 0.04
            - len(trades) * 0.00005
        )
        rows.append(
            {
                "fast_window": int(fast_window),
                "slow_window": int(slow_window),
                "deep_pullback_pct": float(pullback_pct),
                "total_return": float(metrics["total_return"]),
                "annualized_return": float(metrics["annualized_return"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "sharpe": float(metrics["sharpe"]),
                "trades": int(len(trades)),
                "latest_target_position": float(latest["target_position"]),
                "score": float(score),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
