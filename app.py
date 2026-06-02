from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
from gold_advisor.evaluation import evaluate_forward_returns, run_parameter_grid
from gold_advisor.market import (
    LondonGoldConversion,
    convert_cny_per_gram_to_london_gold,
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


@st.cache_data(show_spinner=False, ttl=900)
def cached_forward_evaluation(prices: pd.DataFrame, config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    return evaluate_forward_returns(prices, config)


@st.cache_data(show_spinner=False, ttl=900)
def cached_parameter_grid(prices: pd.DataFrame, config: StrategyConfig, initial_cash: float) -> pd.DataFrame:
    return run_parameter_grid(prices, config, initial_cash=initial_cash)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def metric_value(comparison: pd.DataFrame, metric: str, column: str) -> float:
    return float(comparison.loc[comparison["metric"] == metric, column].iloc[0])


def refresh_key_for_interval(refresh_seconds: int | None) -> int:
    if refresh_seconds is None:
        return 0
    return int(time.time() // refresh_seconds)


def horizon_label(days: int) -> str:
    labels = {1: "1日", 5: "1周", 20: "1月", 60: "1季"}
    return labels.get(days, f"{days}日")


def format_percent_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].map(lambda value: pct(float(value)) if pd.notna(value) else "-")
    return result


def render_tradingview_widget(symbol: str, height: int = 760) -> None:
    config = {
        "autosize": True,
        "symbol": symbol,
        "interval": "60",
        "timezone": "Asia/Shanghai",
        "theme": "light",
        "style": "1",
        "locale": "zh_CN",
        "allow_symbol_change": True,
        "calendar": False,
        "height": height,
        "support_host": "https://www.tradingview.com",
    }
    html = f"""
    <style>
      html, body {{
        height: 100%;
        margin: 0;
        overflow: hidden;
      }}
      .tradingview-widget-container {{
        height: {height}px;
        min-height: {height}px;
        width: 100%;
      }}
      .tradingview-widget-container__widget {{
        height: 100%;
        min-height: {height}px;
        width: 100%;
      }}
    </style>
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {json.dumps(config, ensure_ascii=False)}
      </script>
    </div>
    """
    components.html(html, height=height + 32, scrolling=False)


def build_kline_figure(prices: pd.DataFrame, strategy_equity: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    chart_data = prices.merge(
        strategy_equity[["date", "ma_fast", "ma_slow"]],
        on="date",
        how="left",
    )
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart_data["date"],
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            name="K线",
            increasing_line_color="#059669",
            decreasing_line_color="#dc2626",
        )
    )
    fig.add_trace(go.Scatter(x=chart_data["date"], y=chart_data["ma_fast"], name="短均线", line=dict(color="#0ea5e9", width=1)))
    fig.add_trace(go.Scatter(x=chart_data["date"], y=chart_data["ma_slow"], name="长均线", line=dict(color="#f59e0b", width=1)))

    if not trades.empty:
        buy_trades = trades.loc[trades["side"] == "买入"]
        sell_trades = trades.loc[trades["side"] == "卖出"]
        if not buy_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_trades["date"],
                    y=buy_trades["price"],
                    mode="markers",
                    name="买入点",
                    marker=dict(symbol="triangle-up", color="#16a34a", size=10),
                )
            )
        if not sell_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_trades["date"],
                    y=sell_trades["price"],
                    mode="markers",
                    name="卖出点",
                    marker=dict(symbol="triangle-down", color="#dc2626", size=10),
                )
            )

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=35, b=10),
        title="国内金价日 K 线与模拟调仓点",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    return fig


def build_price_figure(strategy_equity: pd.DataFrame, fast_window: int, slow_window: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy_equity["date"], y=strategy_equity["close"], name="价格", line=dict(color="#1f2937", width=1.8)))
    fig.add_trace(go.Scatter(x=strategy_equity["date"], y=strategy_equity["ma_fast"], name=f"MA{fast_window}", line=dict(color="#0ea5e9", width=1)))
    fig.add_trace(go.Scatter(x=strategy_equity["date"], y=strategy_equity["ma_slow"], name=f"MA{slow_window}", line=dict(color="#f59e0b", width=1)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10), title="价格与均线", hovermode="x unified")
    return fig


with st.sidebar:
    st.header("参数")
    source_label = st.radio("数据源", ["演示数据", "AKShare/上金所", "CSV"], index=1, horizontal=False)
    source = {"演示数据": "demo", "AKShare/上金所": "akshare", "CSV": "csv"}[source_label]
    symbol = st.text_input("上金所品种", value="Au99.99")
    use_delayed_quote = st.checkbox("使用上金所延时价更新当日", value=source == "akshare", disabled=source != "akshare")
    refresh_label = st.selectbox("自动刷新", ["关闭", "30 秒", "1 分钟", "5 分钟", "15 分钟"], index=2)
    refresh_seconds = {"关闭": None, "30 秒": 30, "1 分钟": 60, "5 分钟": 300, "15 分钟": 900}[refresh_label]
    show_external_live_chart = st.checkbox("显示伦敦金实时图", value=True)
    if st.button("刷新数据", width="stretch"):
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

    london_conversion: LondonGoldConversion | None = None
    london_note = "伦敦金折算价暂不可用。"
    try:
        london_conversion = cached_london_conversion(data_refresh_key)
        london_note = f"伦敦金折算价来自 {london_conversion.source}。"
    except Exception as exc:
        london_note = f"伦敦金接口暂不可用。原因：{exc}"

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
    signal_events, signal_summary = cached_forward_evaluation(prices, config)

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

    overview_tab, chart_tab, evaluation_tab, trades_tab = st.tabs(["总览", "行情图", "模型评估", "交易记录"])

    with overview_tab:
        if london_conversion is not None:
            conversion_rows = [
                {"项目": "XAU/USD", "数值": f"{london_conversion.gold_usd_per_oz:.2f} 美元/盎司"},
                {"项目": "USD/CNY", "数值": f"{london_conversion.usd_cny:.4f}"},
                {"项目": "伦敦折人民币", "数值": f"{london_conversion.cny_per_gram:.2f} 元/克"},
                {"项目": "国内价差", "数值": f"{domestic_price - london_conversion.cny_per_gram:.2f} 元/克"},
                {"项目": "国内溢价率", "数值": pct(premium) if premium is not None else "-"},
            ]
            st.dataframe(pd.DataFrame(conversion_rows), hide_index=True, width="stretch")

            with st.expander("伦敦金 / 国内金换算工具"):
                tool_mode = st.radio("方向", ["伦敦金 -> 元/克", "元/克 -> 伦敦金"], horizontal=True)
                tool_cols = st.columns(3)
                fx_for_tool = tool_cols[0].number_input(
                    "USD/CNY",
                    min_value=0.0001,
                    value=float(london_conversion.usd_cny),
                    step=0.001,
                    format="%.4f",
                    key="converter_usd_cny",
                )
                if tool_mode == "伦敦金 -> 元/克":
                    tool_gold_usd = tool_cols[1].number_input(
                        "XAU/USD(美元/盎司)",
                        min_value=0.01,
                        value=float(london_conversion.gold_usd_per_oz),
                        step=1.0,
                        format="%.2f",
                        key="converter_xauusd",
                    )
                    converted_price = convert_london_gold_to_cny_per_gram(tool_gold_usd, fx_for_tool)
                    tool_cols[2].metric("折合元/克", f"{converted_price:.2f}")
                else:
                    tool_cny_price = tool_cols[1].number_input(
                        "国内/银行金(元/克)",
                        min_value=0.01,
                        value=float(domestic_price),
                        step=0.1,
                        format="%.2f",
                        key="converter_cny_g",
                    )
                    converted_gold = convert_cny_per_gram_to_london_gold(tool_cny_price, fx_for_tool)
                    tool_cols[2].metric("折合XAU/USD", f"{converted_gold:.2f}")
                st.caption("换算工具只做单位和汇率换算，不会覆盖策略使用的行情数据。银行实际买卖价仍以银行 App 报价为准。")

        left, right = st.columns([1.2, 1])
        with left:
            equity_fig = go.Figure()
            equity_fig.add_trace(go.Scatter(x=strategy_equity["date"], y=strategy_equity["equity"], name="策略", line=dict(color="#059669", width=1.8)))
            equity_fig.add_trace(go.Scatter(x=benchmark["date"], y=benchmark["equity"], name="买入持有", line=dict(color="#6b7280", width=1.4)))
            equity_fig.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10), title="资金曲线", hovermode="x unified")
            st.plotly_chart(equity_fig, width="stretch")

        with right:
            position_fig = go.Figure()
            position_fig.add_trace(go.Scatter(x=strategy_equity["date"], y=strategy_equity["position"], name="实际仓位", fill="tozeroy", line=dict(color="#2563eb", width=1.3)))
            position_fig.add_trace(go.Scatter(x=strategy_equity["date"], y=strategy_equity["target_position"], name="目标仓位", line=dict(color="#dc2626", width=1)))
            position_fig.update_yaxes(tickformat=".0%")
            position_fig.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10), title="仓位", hovermode="x unified")
            st.plotly_chart(position_fig, width="stretch")

    with chart_tab:
        chart_mode = st.radio("图表", ["国内K线", "伦敦金实时图"], horizontal=True)
        if chart_mode == "国内K线":
            st.plotly_chart(build_kline_figure(prices, strategy_equity, trades), width="stretch")
            st.plotly_chart(build_price_figure(strategy_equity, fast_window, slow_window), width="stretch")
            st.caption("国内图表使用上金所历史日线和延时价，不是秒级 tick 行情。")
        elif show_external_live_chart:
            live_symbol_options = {
                "OANDA:XAUUSD": "伦敦金 XAU/USD",
                "FX_IDC:USDCNY": "美元人民币 USD/CNY",
                "COMEX:GC1!": "COMEX 黄金连续合约",
                "SHFE:AU1!": "沪金期货连续合约",
            }
            live_symbol = st.selectbox(
                "实时图品种",
                options=list(live_symbol_options.keys()),
                format_func=lambda value: live_symbol_options[value],
            )
            live_chart_height = st.slider("实时图高度", min_value=560, max_value=980, value=760, step=20)
            render_tradingview_widget(live_symbol, height=live_chart_height)
            st.caption("外部实时图由 TradingView Widget 加载，和策略回测使用的数据源分开。国内银行积存金实际买卖价仍以银行 App 为准。")
        else:
            st.info("已在侧边栏关闭伦敦金实时图。")

    with evaluation_tab:
        metrics_table = comparison.copy()
        for column in ["strategy", "buy_hold"]:
            metrics_table[column] = metrics_table.apply(
                lambda row: f"{row[column]:.2f}" if row["metric"] == "夏普比率" else pct(float(row[column])),
                axis=1,
            )
        metrics_table = metrics_table.rename(columns={"metric": "指标", "strategy": "策略", "buy_hold": "买入持有"})

        table_left, table_right = st.columns([0.85, 1.15])
        with table_left:
            st.subheader("回测指标")
            st.dataframe(metrics_table, hide_index=True, width="stretch")

        with table_right:
            st.subheader("当前判断")
            st.write(str(latest["reason"]))
            latest_factors = pd.DataFrame(
                [
                    {"因子": "RSI", "数值": f"{latest['rsi']:.1f}"},
                    {"因子": "阶段回撤", "数值": pct(float(latest["pullback"]))},
                    {"因子": "近5日动量", "数值": pct(float(latest["momentum_5d"])) if pd.notna(latest["momentum_5d"]) else "-"},
                    {"因子": "短均线5日斜率", "数值": pct(float(latest["ma_fast_slope_5d"])) if pd.notna(latest["ma_fast_slope_5d"]) else "-"},
                    {"因子": "年化波动", "数值": pct(float(latest["volatility"])) if pd.notna(latest["volatility"]) else "-"},
                    {"因子": "持仓克数", "数值": f"{latest['grams']:.2f} 克"},
                    {"因子": "剩余现金", "数值": f"{latest['cash']:.2f} 元"},
                ]
            )
            st.dataframe(latest_factors, hide_index=True, width="stretch")

        st.subheader("信号事后验证")
        if signal_summary.empty:
            st.info("当前数据不足，暂时无法计算信号事后表现。")
        else:
            signal_display = signal_summary.copy()
            signal_display["观察周期"] = signal_display["horizon_days"].map(horizon_label)
            signal_display = signal_display[
                ["signal", "观察周期", "samples", "avg_return", "median_return", "win_rate", "best_return", "worst_return"]
            ]
            signal_display = signal_display.rename(
                columns={
                    "signal": "信号",
                    "samples": "样本数",
                    "avg_return": "平均收益",
                    "median_return": "中位收益",
                    "win_rate": "上涨胜率",
                    "best_return": "最好",
                    "worst_return": "最差",
                }
            )
            signal_display = format_percent_columns(signal_display, ["平均收益", "中位收益", "上涨胜率", "最好", "最差"])
            st.dataframe(signal_display, hide_index=True, width="stretch")
            st.caption("这里的验证方式是：在某一天只使用当天及以前数据生成信号，再观察之后 1日、1周、1月、1季 的价格变化。")

        st.subheader("参数组合试跑")
        run_grid_search = st.button("运行参数组合试跑", width="stretch")
        if not run_grid_search:
            st.info("参数组合试跑比较耗时，点击按钮后再运行，避免打开页面时长时间空白。")
        else:
            parameter_grid = cached_parameter_grid(prices, config, float(initial_cash))
            if parameter_grid.empty:
                st.info("当前参数范围没有可用组合。")
            else:
                grid_display = parameter_grid.head(12).copy()
                grid_display = grid_display.rename(
                    columns={
                        "fast_window": "短均线",
                        "slow_window": "长均线",
                        "deep_pullback_pct": "回撤阈值",
                        "total_return": "累计收益",
                        "annualized_return": "年化收益",
                        "max_drawdown": "最大回撤",
                        "sharpe": "夏普",
                        "trades": "交易次数",
                        "latest_target_position": "最新目标仓位",
                        "score": "综合分",
                    }
                )
                grid_display = format_percent_columns(grid_display, ["回撤阈值", "累计收益", "年化收益", "最大回撤", "最新目标仓位"])
                grid_display["夏普"] = grid_display["夏普"].map(lambda value: f"{float(value):.2f}")
                grid_display["综合分"] = grid_display["综合分"].map(lambda value: f"{float(value):.4f}")
                st.dataframe(grid_display, hide_index=True, width="stretch")
                st.caption("综合分用于排序，优先考虑年化收益、回撤、夏普和交易次数；它不是收益承诺，只是用来筛选值得继续观察的参数。")

        st.subheader("最近信号")
        recent_signal_events = signal_events.tail(15).copy()
        recent_signal_events["date"] = pd.to_datetime(recent_signal_events["date"]).dt.date
        recent_signal_events["momentum_5d_display"] = recent_signal_events["momentum_5d"].map(
            lambda value: pct(float(value)) if pd.notna(value) else "-"
        )
        for horizon in [1, 5, 20, 60]:
            column = f"return_{horizon}d"
            if column in recent_signal_events.columns:
                recent_signal_events[column] = recent_signal_events[column].map(lambda value: pct(float(value)) if pd.notna(value) else "-")
        recent_signal_events = recent_signal_events[
            [
                "date",
                "close",
                "signal",
                "target_position",
                "momentum_5d_display",
                "return_1d",
                "return_5d",
                "return_20d",
                "return_60d",
                "reason",
            ]
        ].rename(
            columns={
                "date": "日期",
                "close": "价格",
                "signal": "信号",
                "target_position": "目标仓位",
                "momentum_5d_display": "近5日动量",
                "return_1d": "后1日",
                "return_5d": "后1周",
                "return_20d": "后1月",
                "return_60d": "后1季",
                "reason": "判断原因",
            }
        )
        recent_signal_events["价格"] = recent_signal_events["价格"].map(lambda value: f"{float(value):.2f}")
        recent_signal_events["目标仓位"] = recent_signal_events["目标仓位"].map(lambda value: pct(float(value)))
        st.dataframe(recent_signal_events, hide_index=True, width="stretch")

    with trades_tab:
        if not trades.empty:
            recent_trades = trades.tail(30).copy()
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
            st.dataframe(recent_trades, hide_index=True, width="stretch")
        else:
            st.info("当前参数下没有触发模拟调仓。")


render_dashboard()
