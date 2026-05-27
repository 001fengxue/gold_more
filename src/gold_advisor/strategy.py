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


def _target_for_row(row: pd.Series, config: StrategyConfig) -> tuple[str, float, str]:
    if pd.isna(row["ma_slow"]) or pd.isna(row["ma_fast"]):
        return "观望", config.min_position, "长周期指标尚未形成，先保持低仓位。"

    trend_up = row["ma_fast"] > row["ma_slow"]
    deep_pullback = row["pullback"] <= -config.deep_pullback_pct
    cooled_down = row["rsi"] <= config.buy_rsi
    overheated = row["rsi"] >= config.heat_rsi and row["distance_to_fast_ma"] > 0.05

    if overheated:
        return "减仓", config.reduce_position, "RSI 偏高且价格明显高于短均线，优先锁定风险。"
    if trend_up and deep_pullback and cooled_down:
        return "积极买入", config.max_position, "长期趋势仍在，阶段回撤释放了部分风险。"
    if trend_up and cooled_down:
        return "小额买入", min(config.max_position, config.neutral_position + 0.15), "趋势偏强且动量降温，适合增加积存。"
    if trend_up:
        return "持有", config.neutral_position, "价格处于上行趋势，但买入性价比一般。"
    if deep_pullback and cooled_down:
        return "试探买入", config.neutral_position, "趋势偏弱但回撤较深，只适合分批试探。"
    return "防守观望", config.min_position, "趋势偏弱，先控制仓位等待更清晰的价格结构。"


def generate_signals(indicator_frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    records = []
    for _, row in indicator_frame.iterrows():
        signal, target_position, reason = _target_for_row(row, config)
        records.append(
            {
                "date": row["date"],
                "close": row["close"],
                "signal": signal,
                "target_position": target_position,
                "reason": reason,
                "rsi": row["rsi"],
                "pullback": row["pullback"],
                "ma_fast": row["ma_fast"],
                "ma_slow": row["ma_slow"],
            }
        )
    return pd.DataFrame(records)
