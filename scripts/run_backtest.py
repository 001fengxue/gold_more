from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gold_advisor.backtest import compare_metrics, run_buy_hold_benchmark, run_strategy_backtest
from gold_advisor.data import apply_latest_quote, get_sge_delayed_quote, load_prices
from gold_advisor.profiles import PROFILE_ORDER, apply_profile
from gold_advisor.strategy import StrategyConfig


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gold accumulation backtest.")
    parser.add_argument("--source", choices=["demo", "akshare", "csv"], default="demo")
    parser.add_argument("--symbol", default="Au99.99")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--spread-bps", type=float, default=35)
    parser.add_argument("--profile", choices=PROFILE_ORDER, default="收益优先")
    parser.add_argument("--overlay-delayed-quote", action="store_true")
    args = parser.parse_args()

    prices, info = load_prices(args.source, symbol=args.symbol, csv_file=args.csv)
    if args.overlay_delayed_quote and args.source == "akshare":
        prices = apply_latest_quote(prices, get_sge_delayed_quote(args.symbol))
    config = apply_profile(StrategyConfig(spread_bps=args.spread_bps), args.profile)
    strategy_equity, trades, metrics = run_strategy_backtest(prices, config, initial_cash=args.initial_cash)
    benchmark = run_buy_hold_benchmark(prices, spread_bps=args.spread_bps, initial_cash=args.initial_cash)
    comparison = compare_metrics(strategy_equity, benchmark)

    print(f"数据源: {info.name} - {info.description}")
    print(f"策略风格: {args.profile}")
    print(f"区间: {prices['date'].iloc[0].date()} 至 {prices['date'].iloc[-1].date()}")
    latest = strategy_equity.iloc[-1]
    print(f"今日动作: {latest['action']} ({latest['buy_scale']:.2f}x)")
    print(f"持仓状态: {latest['signal']}")
    print(f"目标仓位: {strategy_equity.iloc[-1]['target_position'] * 100:.0f}%")
    print(f"交易次数: {int(metrics['trades'])}")
    print()
    for _, row in comparison.iterrows():
        if row["metric"] == "夏普比率":
            print(f"{row['metric']}: 策略 {row['strategy']:.2f} / 买入持有 {row['buy_hold']:.2f}")
        else:
            print(f"{row['metric']}: 策略 {_format_percent(row['strategy'])} / 买入持有 {_format_percent(row['buy_hold'])}")

    if not trades.empty:
        print()
        print("最近 5 笔调仓:")
        print(trades.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
