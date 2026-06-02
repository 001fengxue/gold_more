from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    fast_window: int = 20
    slow_window: int = 120
    rsi_window: int = 14
    drawdown_window: int = 252
    volatility_window: int = 20
    deep_pullback_pct: float = 0.08
    buy_rsi: float = 55
    heat_rsi: float = 72
    min_position: float = 0.10
    neutral_position: float = 0.35
    max_position: float = 0.75
    reduce_position: float = 0.20
    rebalance_threshold: float = 0.05
    spread_bps: float = 35


def _state_for_row(row: pd.Series, config: StrategyConfig) -> tuple[str, float, str]:
    if pd.isna(row["ma_slow"]) or pd.isna(row["ma_fast"]):
        return "观望", config.min_position, "长周期指标尚未形成，先保持低仓位。"

    fast_slope = row.get("ma_fast_slope_5d", 0)
    momentum_3d = row.get("momentum_3d", 0)
    momentum_5d = row.get("momentum_5d", 0)
    momentum_10d = row.get("momentum_10d", 0)
    down_days_5 = row.get("down_days_5", 0)

    trend_up = row["ma_fast"] > row["ma_slow"] and fast_slope >= -0.001
    deep_pullback = row["pullback"] <= -config.deep_pullback_pct
    cooled_down = row["rsi"] <= config.buy_rsi
    overheated = row["rsi"] >= config.heat_rsi and row["distance_to_fast_ma"] > 0.05
    falling_risk = (
        momentum_5d <= -0.025
        or momentum_10d <= -0.020
        or down_days_5 >= 4
        or (row["distance_to_fast_ma"] <= -0.025 and fast_slope < -0.003)
    )
    stabilizing = momentum_3d > 0 and momentum_5d > -0.020 and down_days_5 <= 3
    yield_first = config.min_position >= 0.70

    if overheated:
        return "减仓", config.reduce_position, "RSI 偏高且价格明显高于短均线，优先锁定风险。"
    if deep_pullback and falling_risk:
        if yield_first:
            return "核心持有", config.min_position, "收益优先模式保留高底仓；短线仍弱，暂不继续追加。"
        return "等待企稳", config.min_position, "价格仍处在短期下跌结构里，先把便宜和可买分开，等待企稳信号。"
    if trend_up and deep_pullback and cooled_down:
        if yield_first:
            return "积极加仓", config.max_position, "长期趋势仍在且短期跌势放缓，收益优先模式提高到进攻仓位。"
        return "小额买入", min(config.max_position, config.neutral_position + 0.15), "长期趋势仍在且短期跌势放缓，可以小额分批增加仓位。"
    if trend_up and cooled_down:
        return "小额买入", min(config.max_position, config.neutral_position + 0.15), "趋势偏强且动量降温，适合分批增加积存。"
    if trend_up:
        if yield_first:
            return "趋势持有", config.neutral_position, "价格处于上行趋势，收益优先模式维持较高仓位。"
        return "持有", config.neutral_position, "价格处于上行趋势，但买入性价比一般。"
    if deep_pullback and cooled_down and stabilizing:
        if yield_first:
            return "核心持有", config.min_position, "价格有初步反弹但趋势尚未修复，保留核心仓位等待确认。"
        return "低仓观察", config.min_position, "价格已有初步反弹，但趋势尚未修复，先观察，不因反弹直接加仓。"
    if deep_pullback and cooled_down:
        if yield_first:
            return "核心持有", config.min_position, "回撤较深但趋势仍弱，收益优先模式保留核心仓位。"
        return "等待企稳", config.min_position, "回撤较深但趋势仍弱，暂不把下跌中的低价直接当成买点。"
    return "防守观望", config.min_position, "趋势偏弱，先控制仓位等待更清晰的价格结构。"


def _action_for_row(row: pd.Series, config: StrategyConfig) -> tuple[str, float, str]:
    if pd.isna(row["ma_slow"]) or pd.isna(row["ma_fast"]):
        return "暂停买入", 0.0, "长周期指标尚未形成，暂不新增买入。"

    fast_slope = row.get("ma_fast_slope_5d", 0)
    momentum_3d = row.get("momentum_3d", 0)
    momentum_5d = row.get("momentum_5d", 0)
    momentum_10d = row.get("momentum_10d", 0)
    down_days_5 = row.get("down_days_5", 0)
    daily_return = row.get("return", 0)

    trend_up = row["ma_fast"] > row["ma_slow"] and fast_slope >= -0.001
    deep_pullback = row["pullback"] <= -config.deep_pullback_pct
    cooled_down = row["rsi"] <= config.buy_rsi
    overheated = row["rsi"] >= config.heat_rsi and row["distance_to_fast_ma"] > 0.05
    sharp_fall = momentum_5d <= -0.025 or momentum_10d <= -0.040 or down_days_5 >= 4
    rebound_probe = daily_return >= 0.015 or (momentum_3d > 0.010 and down_days_5 <= 3)

    if overheated:
        return "暂停买入", 0.0, "价格偏热，暂停新增买入；已有仓位按目标仓位管理。"
    if sharp_fall:
        return "暂停买入", 0.0, "短线下跌仍重，先不接飞刀，等止跌或放量反弹。"
    if trend_up and deep_pullback and cooled_down:
        return "加倍买入", 1.5, "长期趋势仍在，回撤后跌势放缓，可比平时多买。"
    if trend_up and cooled_down:
        return "正常买入", 1.0, "趋势偏强且动量降温，按计划正常买入。"
    if deep_pullback and rebound_probe:
        return "小额试探", 0.25, "大跌后出现反弹，但趋势还没修复，只适合小额试探。"
    if deep_pullback:
        return "暂不加仓", 0.0, "价格便宜但结构未稳，保留核心仓，不新增买入。"
    if trend_up:
        return "小额定投", 0.5, "趋势仍在但性价比一般，只做小额定投。"
    return "暂停买入", 0.0, "趋势偏弱，暂停新增买入。"


def generate_signals(indicator_frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    records = []
    for _, row in indicator_frame.iterrows():
        signal, target_position, reason = _state_for_row(row, config)
        action, buy_scale, action_reason = _action_for_row(row, config)
        records.append(
            {
                "date": row["date"],
                "close": row["close"],
                "signal": signal,
                "target_position": target_position,
                "reason": reason,
                "action": action,
                "buy_scale": buy_scale,
                "action_reason": action_reason,
                "rsi": row["rsi"],
                "pullback": row["pullback"],
                "ma_fast": row["ma_fast"],
                "ma_slow": row["ma_slow"],
                "momentum_3d": row.get("momentum_3d"),
                "momentum_5d": row.get("momentum_5d"),
                "momentum_10d": row.get("momentum_10d"),
                "ma_fast_slope_5d": row.get("ma_fast_slope_5d"),
                "down_days_5": row.get("down_days_5"),
            }
        )
    return pd.DataFrame(records)
