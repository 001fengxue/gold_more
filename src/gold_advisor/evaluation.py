from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Iterable

import pandas as pd

from .backtest import compute_metrics, run_buy_hold_benchmark, run_strategy_backtest
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
    signal_columns = ["date", "signal", "target_position", "reason", "action", "buy_scale", "action_reason"]
    data = indicators.merge(signals[signal_columns], on="date", how="left")

    for horizon in horizon_days:
        data[f"return_{horizon}d"] = data["close"].shift(-horizon) / data["close"] - 1

    summary_rows: list[dict[str, float | int | str]] = []
    for action, group in data.groupby("action", sort=False):
        for horizon in horizon_days:
            returns = group[f"return_{horizon}d"].dropna().astype(float)
            if returns.empty:
                continue
            summary_rows.append(
                {
                    "action": action,
                    "avg_buy_scale": float(group["buy_scale"].mean()),
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


def _simulate_walk_forward_targets(
    signal_frame: pd.DataFrame,
    config: StrategyConfig,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    cash = initial_cash
    grams = 0.0
    spread = config.spread_bps / 10_000
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    trades: list[dict[str, float | str | pd.Timestamp]] = []

    for _, row in signal_frame.iterrows():
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
                **row.to_dict(),
                "cash": cash,
                "grams": grams,
                "equity": equity,
                "position": 0.0 if equity <= 0 else grams * close / equity,
            }
        )

    equity_frame = pd.DataFrame(rows)
    trade_frame = pd.DataFrame(trades)
    metrics = compute_metrics(equity_frame) if not equity_frame.empty else {}
    metrics["trades"] = float(len(trade_frame))
    metrics["final_cash"] = float(cash)
    metrics["final_grams"] = float(grams)
    return equity_frame, trade_frame, metrics


def run_walk_forward_validation(
    prices: pd.DataFrame,
    base_config: StrategyConfig,
    initial_cash: float,
    train_window: int = 756,
    validation_window: int = 20,
    max_validation_rows: int = 126,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Train parameters only on past data, then validate them on later rows."""
    if len(prices) < train_window + validation_window:
        raise ValueError("数据长度不足，无法进行滚动样本外验证。")

    validation_start = max(train_window, len(prices) - max_validation_rows)
    rows: list[pd.DataFrame] = []
    period_rows: list[dict[str, float | int | str | pd.Timestamp]] = []

    for start in range(validation_start, len(prices), validation_window):
        end = min(start + validation_window, len(prices))
        train_start = max(0, start - train_window)
        train_prices = prices.iloc[train_start:start].reset_index(drop=True)
        if len(train_prices) < train_window:
            continue

        grid = run_parameter_grid(train_prices, base_config, initial_cash=initial_cash)
        if grid.empty:
            continue

        best = grid.iloc[0]
        trained_config = replace(
            base_config,
            fast_window=int(best["fast_window"]),
            slow_window=int(best["slow_window"]),
            deep_pullback_pct=float(best["deep_pullback_pct"]),
        )

        context = prices.iloc[:end].reset_index(drop=True)
        indicators = add_indicators(
            context,
            fast_window=trained_config.fast_window,
            slow_window=trained_config.slow_window,
            rsi_window=trained_config.rsi_window,
            drawdown_window=trained_config.drawdown_window,
            volatility_window=trained_config.volatility_window,
        )
        signals = generate_signals(indicators, trained_config)
        signal_columns = ["date", "signal", "target_position", "reason", "action", "buy_scale", "action_reason"]
        validation = indicators.merge(signals[signal_columns], on="date", how="left").iloc[start:end].copy()

        validation["train_start"] = train_prices["date"].iloc[0]
        validation["train_end"] = train_prices["date"].iloc[-1]
        validation["validation_start"] = prices["date"].iloc[start]
        validation["validation_end"] = prices["date"].iloc[end - 1]
        validation["trained_fast_window"] = trained_config.fast_window
        validation["trained_slow_window"] = trained_config.slow_window
        validation["trained_pullback_pct"] = trained_config.deep_pullback_pct
        validation["train_score"] = float(best["score"])
        validation["train_total_return"] = float(best["total_return"])
        validation["train_max_drawdown"] = float(best["max_drawdown"])
        rows.append(validation)

        period_rows.append(
            {
                "validation_start": prices["date"].iloc[start],
                "validation_end": prices["date"].iloc[end - 1],
                "train_start": train_prices["date"].iloc[0],
                "train_end": train_prices["date"].iloc[-1],
                "fast_window": trained_config.fast_window,
                "slow_window": trained_config.slow_window,
                "deep_pullback_pct": trained_config.deep_pullback_pct,
                "train_score": float(best["score"]),
                "train_total_return": float(best["total_return"]),
                "train_max_drawdown": float(best["max_drawdown"]),
                "train_sharpe": float(best["sharpe"]),
                "validation_rows": int(len(validation)),
            }
        )

    if not rows:
        raise ValueError("没有生成任何样本外验证区间。")

    walk_signals = pd.concat(rows, ignore_index=True)
    for horizon in DEFAULT_FORWARD_HORIZONS:
        walk_signals[f"return_{horizon}d"] = walk_signals["close"].shift(-horizon) / walk_signals["close"] - 1

    equity, trades, metrics = _simulate_walk_forward_targets(walk_signals, base_config, initial_cash)
    benchmark = run_buy_hold_benchmark(
        prices.loc[prices["date"].isin(equity["date"])].reset_index(drop=True),
        spread_bps=base_config.spread_bps,
        initial_cash=initial_cash,
    )
    periods = pd.DataFrame(period_rows)
    return equity, trades, periods, metrics, benchmark
