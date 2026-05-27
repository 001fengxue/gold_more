from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gold_advisor.backtest import compare_metrics, run_buy_hold_benchmark, run_strategy_backtest
from gold_advisor.data import (
    DataSourceInfo,
    SgeDelayedQuote,
    apply_latest_quote,
    generate_demo_prices,
    get_sge_delayed_quote,
    load_csv_prices,
    load_prices,
)
from gold_advisor.market import (
    LondonGoldConversion,
    convert_london_gold_to_cny_per_gram,
    get_london_gold_conversion,
)
from gold_advisor.strategy import StrategyConfig


st.set_page_config(page_title="积存金决策辅助", layout="wide")


@st.cache_data(show_spinner=False, ttl=900)
def cached_load_prices(source: str, symbol: str, csv_bytes: bytes | None) -> tuple[pd.DataFrame, DataSourceInfo]:
    if source == "csv" and csv_bytes is not None:
        from io import BytesIO

        return load_csv_prices(BytesIO(csv_bytes)), DataSourceInfo("CSV", "上传数据")
    if source == "demo":
        return generate_demo_prices(), DataSourceInfo("Demo", "离线演示数据")
    return load_prices(source, symbol=symbol)


@st.cache_data(show_spinner=False, ttl=300)
def cached_delayed_quote(symbol: str, refresh_key: int = 0) -> SgeDelayedQuote:
    del refresh_key
    return get_sge_delayed_quote(symbol)


@st.cache_data(show_spinner=False, ttl=300)
def cached_london_conversion(refresh_key: int = 0) -> LondonGoldConversion:
    del refresh_key
    return get_london_gold_conversion()


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def metric_value(comparison: pd.DataFrame, metric: str, column: str) -> float:
    return float(comparison.loc[comparison["metric"] == metric, column].iloc[0])


def refresh_key_for_interval(refresh_seconds: int | None) -> int:
    if refresh_seconds is None:
        return 0
    return int(time.time() // refresh_seconds)


with st.sidebar:
    st.header("参数")
    source_label = st.radio("数据源", ["演示数据", "AKShare/上金所", "CSV"], index=1, horizontal=False)
    source = {"演示数据": "demo", "AKShare/上金所": "akshare", "CSV": "csv"}[source_label]
    symbol = st.text_input("上金所品种", value="Au99.99")
    use_delayed_quote = st.checkbox("使用上金所延时价更新当日", value=source == "akshare", disabled=source != "akshare")
    manual_quote_enabled = st.checkbox("手动填入银行积存金价")
    manual_quote = st.number_input("银行价(元/克)", min_value=0.0, value=0.0, step=0.1, disabled=not manual_quote_enabled)
    manual_london_enabled = st.checkbox("手动填入伦敦金/汇率")
    manual_gold_usd = st.number_input("XAU/USD(美元/盎司)", min_value=0.0, value=0.0, step=1.0, disabled=not manual_london_enabled)
    manual_usd_cny = st.number_input("USD/CNY", min_value=0.0, value=0.0, step=0.001, format="%.4f", disabled=not manual_london_enabled)
    refresh_label = st.selectbox("自动刷新", ["关闭", "30 秒", "1 分钟", "5 分钟", "15 分钟"], index=2)
    refresh_seconds = {"关闭": None, "30 秒": 30, "1 分钟": 60, "5 分钟": 300, "15 分钟": 900}[refresh_label]
    if st.button("刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    csv_file = st.file_uploader("CSV", type=["csv"]) if source == "csv" else None

    initial_cash = st.number_input("初始资金", min_value=1_000, max_value=10_000_000, value=100_000, step=10_000)
    spread_bps = st.slider("买卖价差 bps", min_value=0, max_value=200, value=35, step=5)
    fast_window = st.slider("短均线", min_value=5, max_value=60, value=20, step=5)
    slow_window = st.slider("长均线", min_value=60, max_value=260, value=120, step=10)
    deep_pullback_pct = st.slider("深度回撤", min_value=0.03, max_value=0.20, value=0.08, step=0.01)
    rebalance_threshold = st.slider("调仓阈值", min_value=0.01, max_value=0.20, value=0.05, step=0.01)


@st.fragment(run_every=refresh_seconds)
def render_dashboard() -> None:
    st.title("积存金决策辅助系统")

    csv_bytes = csv_file.getvalue() if csv_file is not None else None
    data_refresh_key = refresh_key_for_interval(refresh_seconds)
    try:
        prices, info = cached_load_prices(source, symbol, csv_bytes)
    except Exception as exc:
        st.warning(f"数据源暂不可用，已切换到演示数据。原因：{exc}")
        prices, info = generate_demo_prices(), DataSourceInfo("Demo", "离线演示数据")

    quote_note = "当前价格来自历史日线收盘价。"
    if source == "akshare" and use_delayed_quote:
        try:
            delayed_quote = cached_delayed_quote(symbol, data_refresh_key)
            prices = apply_latest_quote(prices, delayed_quote)
            quote_day = delayed_quote.quote_date.date() if delayed_quote.quote_date is not None else "今天"
            quote_note = f"当前价格已用上金所{quote_day}延时行情更新，合约 {delayed_quote.symbol}。"
        except Exception as exc:
            quote_note = f"上金所延时行情暂不可用，仍使用历史日线收盘价。原因：{exc}"

    if manual_quote_enabled and manual_quote > 0:
        manual = SgeDelayedQuote(
            symbol="银行积存金手动价",
            last=float(manual_quote),
            high=float(manual_quote),
            low=float(manual_quote),
            open=float(manual_quote),
            quote_date=prices["date"].iloc[-1],
            source="手动输入",
        )
        prices = apply_latest_quote(prices, manual)
        quote_note = f"当前价格已按手动银行积存金价 {manual_quote:.2f} 元/克校准。"

    london_conversion: LondonGoldConversion | None = None
    london_note = "伦敦金折算价暂不可用。"
    if manual_london_enabled and manual_gold_usd > 0 and manual_usd_cny > 0:
        london_conversion = LondonGoldConversion(
            gold_usd_per_oz=float(manual_gold_usd),
            usd_cny=float(manual_usd_cny),
            cny_per_gram=convert_london_gold_to_cny_per_gram(float(manual_gold_usd), float(manual_usd_cny)),
            gold_quote_time=None,
            fx_quote_time=None,
            source="手动输入",
        )
        london_note = "伦敦金折算价来自手动输入。"
    elif not manual_london_enabled:
        try:
            london_conversion = cached_london_conversion(data_refresh_key)
            london_note = f"伦敦金折算价来自 {london_conversion.source}。"
        except Exception as exc:
            london_note = f"伦敦金接口暂不可用，可改用手动输入。原因：{exc}"

    config = StrategyConfig(
        fast_window=fast_window,
        slow_window=slow_window,
        deep_pullback_pct=deep_pullback_pct,
        spread_bps=spread_bps,
        rebalance_threshold=rebalance_threshold,
    )

    strategy_equity, trades, metrics = run_strategy_backtest(prices, config, initial_cash=float(initial_cash))
    benchmark = run_buy_hold_benchmark(prices, spread_bps=spread_bps, initial_cash=float(initial_cash))
    comparison = compare_metrics(strategy_equity, benchmark)
    latest = strategy_equity.iloc[-1]
    domestic_price = float(latest["close"])
    london_price = london_conversion.cny_per_gram if london_conversion is not None else None
    premium = domestic_price / london_price - 1 if london_price and london_price > 0 else None

    market_cols = st.columns(3)
    market_cols[0].metric("国内价(元/克)", f"{domestic_price:.2f}")
    market_cols[1].metric("伦敦折算", f"{london_price:.2f}" if london_price is not None else "-")
    market_cols[2].metric("国内溢价", pct(premium) if premium is not None else "-")

    signal_cols = st.columns(4)
    signal_cols[0].metric("当前信号", str(latest["signal"]))
    signal_cols[1].metric("目标仓位", f"{latest['target_position'] * 100:.0f}%")
    signal_cols[2].metric("策略累计收益", pct(metric_value(comparison, "累计收益", "strategy")))
    signal_cols[3].metric("最大回撤", pct(metric_value(comparison, "最大回撤", "strategy")))

    refresh_note = "自动刷新已关闭" if refresh_seconds is None else f"自动刷新：{refresh_label}"
    st.caption(
        f"数据源：{info.name}，区间：{prices['date'].iloc[0].date()} 至 {prices['date'].iloc[-1].date()}。"
        f"{quote_note} {london_note} {refresh_note}。本系统只用于研究和记录，不构成投资建议。"
    )

    if london_conversion is not None:
        conversion_rows = [
            {"项目": "XAU/USD", "数值": f"{london_conversion.gold_usd_per_oz:.2f} 美元/盎司"},
            {"项目": "USD/CNY", "数值": f"{london_conversion.usd_cny:.4f}"},
            {"项目": "伦敦折人民币", "数值": f"{london_conversion.cny_per_gram:.2f} 元/克"},
            {"项目": "国内价差", "数值": f"{domestic_price - london_conversion.cny_per_gram:.2f} 元/克"},
            {"项目": "国内溢价率", "数值": pct(premium) if premium is not None else "-"},
        ]
        st.dataframe(pd.DataFrame(conversion_rows), hide_index=True, use_container_width=True)

    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=strategy_equity["date"], y=strategy_equity["close"], name="价格", line=dict(color="#1f2937", width=1.8)))
    price_fig.add_trace(go.Scatter(x=strategy_equity["date"], y=strategy_equity["ma_fast"], name=f"MA{fast_window}", line=dict(color="#0ea5e9", width=1)))
    price_fig.add_trace(go.Scatter(x=strategy_equity["date"], y=strategy_equity["ma_slow"], name=f"MA{slow_window}", line=dict(color="#f59e0b", width=1)))
    price_fig.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10), title="价格与均线", hovermode="x unified")
    st.plotly_chart(price_fig, use_container_width=True)

    left, right = st.columns([1.2, 1])

    with left:
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(x=strategy_equity["date"], y=strategy_equity["equity"], name="策略", line=dict(color="#059669", width=1.8)))
        equity_fig.add_trace(go.Scatter(x=benchmark["date"], y=benchmark["equity"], name="买入持有", line=dict(color="#6b7280", width=1.4)))
        equity_fig.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10), title="资金曲线", hovermode="x unified")
        st.plotly_chart(equity_fig, use_container_width=True)

    with right:
        position_fig = go.Figure()
        position_fig.add_trace(go.Scatter(x=strategy_equity["date"], y=strategy_equity["position"], name="实际仓位", fill="tozeroy", line=dict(color="#2563eb", width=1.3)))
        position_fig.add_trace(go.Scatter(x=strategy_equity["date"], y=strategy_equity["target_position"], name="目标仓位", line=dict(color="#dc2626", width=1)))
        position_fig.update_yaxes(tickformat=".0%")
        position_fig.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10), title="仓位", hovermode="x unified")
        st.plotly_chart(position_fig, use_container_width=True)

    metrics_table = comparison.copy()
    for column in ["strategy", "buy_hold"]:
        metrics_table[column] = metrics_table.apply(
            lambda row: f"{row[column]:.2f}" if row["metric"] == "夏普比率" else pct(float(row[column])),
            axis=1,
        )
    metrics_table = metrics_table.rename(columns={"metric": "指标", "strategy": "策略", "buy_hold": "买入持有"})

    table_left, table_right = st.columns([0.8, 1.2])
    with table_left:
        st.subheader("评估")
        st.dataframe(metrics_table, hide_index=True, use_container_width=True)

    with table_right:
        st.subheader("今日判断")
        st.write(str(latest["reason"]))
        latest_factors = pd.DataFrame(
            [
                {"因子": "RSI", "数值": f"{latest['rsi']:.1f}"},
                {"因子": "阶段回撤", "数值": pct(float(latest["pullback"]))},
                {"因子": "年化波动", "数值": pct(float(latest["volatility"])) if pd.notna(latest["volatility"]) else "-"},
                {"因子": "持仓克数", "数值": f"{latest['grams']:.2f} 克"},
                {"因子": "剩余现金", "数值": f"{latest['cash']:.2f} 元"},
            ]
        )
        st.dataframe(latest_factors, hide_index=True, use_container_width=True)

    if not trades.empty:
        st.subheader("最近调仓")
        recent_trades = trades.tail(12).copy()
        recent_trades["date"] = pd.to_datetime(recent_trades["date"]).dt.date
        recent_trades["price"] = recent_trades["price"].map(lambda value: f"{value:.2f}")
        recent_trades["grams"] = recent_trades["grams"].map(lambda value: f"{value:.2f}")
        recent_trades["target_position"] = recent_trades["target_position"].map(lambda value: f"{value * 100:.0f}%")
        recent_trades["spread_cost"] = recent_trades["spread_cost"].map(lambda value: f"{value:.2f}")
        recent_trades = recent_trades.rename(
            columns={
                "date": "日期",
                "side": "方向",
                "price": "价格",
                "grams": "克数",
                "signal": "信号",
                "target_position": "目标仓位",
                "spread_cost": "价差成本",
            }
        )
        st.dataframe(recent_trades, hide_index=True, use_container_width=True)


render_dashboard()
