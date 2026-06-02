from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gold_advisor.data import apply_latest_quote, get_sge_delayed_quote, load_prices
from gold_advisor.evaluation import run_walk_forward_validation
from gold_advisor.strategy import StrategyConfig


def _format_percent(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward validation for gold signals.")
    parser.add_argument("--source", choices=["demo", "akshare", "csv"], default="akshare")
    parser.add_argument("--symbol", default="Au99.99")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--spread-bps", type=float, default=35)
    parser.add_argument("--train-window", type=int, default=756)
    parser.add_argument("--validation-window", type=int, default=20)
    parser.add_argument("--max-validation-rows", type=int, default=126)
    parser.add_argument("--overlay-delayed-quote", action="store_true")
    args = parser.parse_args()

    prices, info = load_prices(args.source, symbol=args.symbol, csv_file=args.csv)
    if args.overlay_delayed_quote and args.source == "akshare":
        prices = apply_latest_quote(prices, get_sge_delayed_quote(args.symbol))

    config = StrategyConfig(spread_bps=args.spread_bps)
    equity, trades, periods, metrics, benchmark = run_walk_forward_validation(
        prices,
        config,
        initial_cash=args.initial_cash,
        train_window=args.train_window,
        validation_window=args.validation_window,
        max_validation_rows=args.max_validation_rows,
    )

    print(f"数据源: {info.name} - {info.description}")
    print(f"样本外区间: {equity['date'].iloc[0].date()} 至 {equity['date'].iloc[-1].date()}")
    print(f"训练窗口: {args.train_window} 个交易日; 每段验证: {args.validation_window} 个交易日")
    print()
    print("样本外表现:")
    print(f"累计收益: {_format_percent(metrics['total_return'])}")
    print(f"年化收益: {_format_percent(metrics['annualized_return'])}")
    print(f"最大回撤: {_format_percent(metrics['max_drawdown'])}")
    print(f"夏普比率: {metrics['sharpe']:.2f}")
    print(f"交易次数: {int(metrics['trades'])}")
    print(f"买入持有同期收益: {_format_percent(benchmark['equity'].iloc[-1] / benchmark['equity'].iloc[0] - 1)}")

    print()
    print("每段训练后选出的参数:")
    display_periods = periods.copy()
    for column in ["validation_start", "validation_end", "train_start", "train_end"]:
        display_periods[column] = pd.to_datetime(display_periods[column]).dt.date
    display_periods["deep_pullback_pct"] = display_periods["deep_pullback_pct"].map(_format_percent)
    display_periods["train_total_return"] = display_periods["train_total_return"].map(_format_percent)
    display_periods["train_max_drawdown"] = display_periods["train_max_drawdown"].map(_format_percent)
    display_periods["train_sharpe"] = display_periods["train_sharpe"].map(lambda value: f"{value:.2f}")
    display_periods["train_score"] = display_periods["train_score"].map(lambda value: f"{value:.4f}")
    print(display_periods.to_string(index=False))

    print()
    print("最近一个月样本外明细:")
    last_month = equity[equity["date"] >= equity["date"].max() - pd.Timedelta(days=35)].copy()
    for column in ["date", "validation_start", "validation_end"]:
        last_month[column] = pd.to_datetime(last_month[column]).dt.date
    for column in [
        "momentum_5d",
        "momentum_10d",
        "ma_fast_slope_5d",
        "pullback",
        "target_position",
        "trained_pullback_pct",
        "return_1d",
        "return_5d",
        "return_20d",
    ]:
        if column in last_month.columns:
            last_month[column] = last_month[column].map(_format_percent)
    columns = [
        "date",
        "close",
        "signal",
        "target_position",
        "momentum_5d",
        "momentum_10d",
        "trained_fast_window",
        "trained_slow_window",
        "trained_pullback_pct",
        "return_1d",
        "return_5d",
        "return_20d",
        "reason",
    ]
    print(last_month[columns].to_string(index=False))


if __name__ == "__main__":
    main()
