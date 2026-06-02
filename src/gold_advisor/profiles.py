from __future__ import annotations

from dataclasses import replace

from .strategy import StrategyConfig


PROFILE_ORDER = ("收益优先", "均衡", "风控优先")

PROFILE_DEFAULTS: dict[str, dict[str, float | int]] = {
    "收益优先": {
        "fast_window": 40,
        "slow_window": 200,
        "deep_pullback_pct": 0.12,
        "min_position": 0.95,
        "neutral_position": 1.00,
        "max_position": 1.00,
        "reduce_position": 0.90,
        "rebalance_threshold": 0.10,
    },
    "均衡": {
        "fast_window": 20,
        "slow_window": 120,
        "deep_pullback_pct": 0.08,
        "min_position": 0.35,
        "neutral_position": 0.65,
        "max_position": 0.95,
        "reduce_position": 0.45,
        "rebalance_threshold": 0.08,
    },
    "风控优先": {
        "fast_window": 20,
        "slow_window": 120,
        "deep_pullback_pct": 0.08,
        "min_position": 0.10,
        "neutral_position": 0.35,
        "max_position": 0.75,
        "reduce_position": 0.20,
        "rebalance_threshold": 0.05,
    },
}


def profile_defaults(profile: str) -> dict[str, float | int]:
    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"未知策略风格: {profile}")
    return PROFILE_DEFAULTS[profile].copy()


def apply_profile(
    config: StrategyConfig,
    profile: str,
    *,
    fast_window: int | None = None,
    slow_window: int | None = None,
    deep_pullback_pct: float | None = None,
    rebalance_threshold: float | None = None,
) -> StrategyConfig:
    defaults = profile_defaults(profile)
    if fast_window is not None:
        defaults["fast_window"] = int(fast_window)
    if slow_window is not None:
        defaults["slow_window"] = int(slow_window)
    if deep_pullback_pct is not None:
        defaults["deep_pullback_pct"] = float(deep_pullback_pct)
    if rebalance_threshold is not None:
        defaults["rebalance_threshold"] = float(rebalance_threshold)
    return replace(config, **defaults)
