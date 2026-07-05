from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

VENV_SITE_PACKAGES = Path(__file__).resolve().parents[1] / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

import scanner
import system_signal_backtest


@dataclass(frozen=True)
class RsMetricVariant:
    name: str
    description: str


VARIANTS = [
    RsMetricVariant("excess_20d", "Current: 20d excess return vs SPY/QQQ"),
    RsMetricVariant("relative_ratio_20d", "20d return of the price ratio vs SPY/QQQ"),
    RsMetricVariant("blend_20_60", "Weighted excess return: 60% 20d + 40% 60d"),
    RsMetricVariant("blend_20_60_120", "Weighted excess return: 50% 20d + 30% 60d + 20% 120d"),
]

THRESHOLD_GRID = [
    (0.0, -2.0),
    (0.0, 0.0),
    (2.0, 0.0),
    (5.0, 0.0),
    (8.0, 2.0),
]


def weighted_excess(
    stock_close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    windows: list[tuple[int, float]],
) -> pd.DataFrame:
    result = pd.DataFrame(0.0, index=stock_close.index, columns=stock_close.columns)
    for window, weight in windows:
        stock_ret = stock_close.pct_change(window) * 100
        bench_ret = benchmark_close.pct_change(window) * 100
        result = result.add(weight * stock_ret.sub(bench_ret, axis=0), fill_value=0.0)
    return result


def relative_ratio_return(
    stock_close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    ratio = stock_close.div(benchmark_close, axis=0)
    return ratio.pct_change(window) * 100


def compute_rs_frames(
    close: pd.DataFrame,
    macro_close: pd.DataFrame,
    variant_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spy_close = pd.DataFrame(
        np.repeat(macro_close["SPY"].to_numpy()[:, None], len(close.columns), axis=1),
        index=close.index,
        columns=close.columns,
    )
    qqq_close = pd.DataFrame(
        np.repeat(macro_close["QQQ"].to_numpy()[:, None], len(close.columns), axis=1),
        index=close.index,
        columns=close.columns,
    )

    if variant_name == "excess_20d":
        stock_20d = close.pct_change(20) * 100
        rs_spy = stock_20d.sub((macro_close["SPY"].pct_change(20) * 100), axis=0)
        rs_qqq = stock_20d.sub((macro_close["QQQ"].pct_change(20) * 100), axis=0)
        return rs_spy, rs_qqq
    if variant_name == "relative_ratio_20d":
        return relative_ratio_return(close, spy_close, 20), relative_ratio_return(close, qqq_close, 20)
    if variant_name == "blend_20_60":
        return (
            weighted_excess(close, spy_close, [(20, 0.6), (60, 0.4)]),
            weighted_excess(close, qqq_close, [(20, 0.6), (60, 0.4)]),
        )
    if variant_name == "blend_20_60_120":
        return (
            weighted_excess(close, spy_close, [(20, 0.5), (60, 0.3), (120, 0.2)]),
            weighted_excess(close, qqq_close, [(20, 0.5), (60, 0.3), (120, 0.2)]),
        )
    raise ValueError(f"Unknown variant {variant_name}")


def compute_state_with_rs(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    leveraged_flags: pd.Series,
    macro_state: pd.DataFrame,
    trend_mode: str,
    rs_vs_spy_20d: pd.DataFrame,
    rs_vs_qqq_20d: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()

    trend = scanner.build_trend_mask(close, ma20, ma50, ma150, trend_mode=trend_mode)
    hh50 = close.rolling(50).max()
    prior_hh50 = hh50.shift(1)
    breakout = high >= prior_hh50
    breakout_age = scanner.bars_since_true(breakout)

    close_to_breakout = close >= prior_hh50 * (1 - scanner.SETUP_PROXIMITY)
    pre_breakout = close < prior_hh50
    fresh_breakout = breakout_age <= scanner.MAX_BREAKOUT_AGE

    extension_from_ma20 = close / ma20 - 1
    extension_from_ma50 = close / ma50 - 1
    extension_from_breakout = close / prior_hh50 - 1
    stock_change_20d_pct = close.pct_change(20) * 100
    runup_10d_pct = close.pct_change(10) * 100
    avg_dollar_vol_20d = (close * volume).rolling(20).mean()

    non_leveraged = pd.DataFrame(
        {col: not bool(leveraged_flags.get(col, False)) for col in close.columns},
        index=close.index,
    )
    price_ok = close >= system_signal_backtest.MIN_PRICE
    liquidity_ok = avg_dollar_vol_20d >= system_signal_backtest.MIN_AVG_DOLLAR_VOL
    runup_ok = runup_10d_pct <= scanner.MAX_10D_RUNUP_PCT
    setup_ma50_not_extended = scanner.build_ma50_extension_mask(extension_from_ma50, trend_mode, scanner.SETUP_MAX_ABOVE_MA50_PCT)
    breakout_ma50_not_extended = scanner.build_ma50_extension_mask(extension_from_ma50, trend_mode, scanner.BREAKOUT_MAX_ABOVE_MA50_PCT)
    momentum_ma50_not_extended = scanner.build_ma50_extension_mask(extension_from_ma50, trend_mode, scanner.MOMENTUM_MAX_ABOVE_MA50_PCT)
    leader_ma50_not_extended = scanner.build_ma50_extension_mask(extension_from_ma50, trend_mode, scanner.LEADER_MAX_ABOVE_MA50_PCT)
    pullback_zone = extension_from_ma20 >= -0.02
    controlled_breakout = extension_from_breakout <= scanner.MAX_BREAKOUT_EXTENSION
    guardrails = non_leveraged & price_ok & liquidity_ok & runup_ok

    setup_candidate = (
        trend
        & close_to_breakout
        & pre_breakout
        & (extension_from_ma20 <= (scanner.SETUP_MAX_ABOVE_MA20_PCT / 100))
        & setup_ma50_not_extended
        & pullback_zone
        & guardrails
    )
    breakout_candidate = (
        trend
        & fresh_breakout
        & (close >= prior_hh50)
        & controlled_breakout
        & (extension_from_ma20 <= (scanner.BREAKOUT_MAX_ABOVE_MA20_PCT / 100))
        & breakout_ma50_not_extended
        & pullback_zone
        & guardrails
    )
    momentum_candidate = (
        trend
        & fresh_breakout
        & (close >= prior_hh50)
        & (extension_from_breakout <= scanner.MOMENTUM_MAX_BREAKOUT_EXTENSION)
        & (extension_from_ma20 <= (scanner.MOMENTUM_MAX_ABOVE_MA20_PCT / 100))
        & momentum_ma50_not_extended
        & (runup_10d_pct <= scanner.MOMENTUM_MAX_10D_RUNUP_PCT)
        & non_leveraged
        & price_ok
        & liquidity_ok
    )
    leader_candidate = (
        trend
        & (breakout_age <= scanner.LEADER_MAX_BREAKOUT_AGE)
        & (close >= prior_hh50)
        & (extension_from_breakout >= -(scanner.LEADER_MAX_BELOW_BREAKOUT_PCT / 100))
        & (close > ma20)
        & (extension_from_ma20 >= 0)
        & (extension_from_ma20 <= (scanner.LEADER_MAX_ABOVE_MA20_PCT / 100))
        & leader_ma50_not_extended
        & (runup_10d_pct <= scanner.LEADER_MAX_10D_RUNUP_PCT)
        & non_leveraged
        & price_ok
        & liquidity_ok
    )
    reclaim_candidate = (
        (ma50 >= ma150 * 0.97)
        & (close > ma20)
        & (ma20 >= ma20.shift(3) * 0.995)
        & (close >= close.rolling(10).max().shift(1))
        & (extension_from_ma20 >= 0)
        & (extension_from_ma20 <= (scanner.RECLAIM_MAX_ABOVE_MA20_PCT / 100))
        & setup_ma50_not_extended
        & (runup_10d_pct <= scanner.RECLAIM_MAX_10D_RUNUP_PCT)
        & non_leveraged
        & price_ok
        & liquidity_ok
    )
    momentum_candidate = momentum_candidate & ~(setup_candidate | breakout_candidate)
    leader_candidate = leader_candidate & ~(setup_candidate | breakout_candidate | momentum_candidate)
    reclaim_candidate = reclaim_candidate & ~(setup_candidate | breakout_candidate | momentum_candidate | leader_candidate)
    candidate = setup_candidate | breakout_candidate | momentum_candidate | leader_candidate | reclaim_candidate

    distance_to_breakout = ((prior_hh50 - close) / prior_hh50).clip(lower=0)
    setup_readiness = 1 - (distance_to_breakout / scanner.SETUP_PROXIMITY).clip(lower=0, upper=1)

    prev_close = close.shift(1)
    true_range = pd.DataFrame(
        np.maximum.reduce(
            [
                (high - low).to_numpy(),
                (high - prev_close).abs().to_numpy(),
                (low - prev_close).abs().to_numpy(),
            ]
        ),
        index=close.index,
        columns=close.columns,
    )
    atr20 = true_range.rolling(20).mean()
    atr200 = true_range.rolling(200).mean()
    atr100 = true_range.rolling(100).mean()
    atr60 = true_range.rolling(60).mean()
    atr_baseline = atr200.combine_first(atr100).combine_first(atr60)
    tightness = (1 - atr20 / atr_baseline).clip(0, 1)

    trend_strength = ((ma50 / ma150) - 1).clip(lower=0, upper=0.20) / 0.20
    ma_stretch_score = 1 - (extension_from_ma20.abs() / (scanner.MAX_ABOVE_MA20_PCT / 100)).clip(lower=0, upper=1)
    breakout_extension_score = 1 - (extension_from_breakout.clip(lower=0) / scanner.MAX_BREAKOUT_EXTENSION).clip(lower=0, upper=1)
    freshness_score = 1 - (breakout_age / scanner.MAX_BREAKOUT_AGE).clip(lower=0, upper=1)
    freshness_score = freshness_score.fillna(0)
    signal_type_bonus = (
        setup_candidate.astype(float) * 1.0
        + breakout_candidate.astype(float) * 0.7
        + momentum_candidate.astype(float) * 0.55
        + leader_candidate.astype(float) * 0.5
        + reclaim_candidate.astype(float) * 0.45
    )
    signal_type = pd.DataFrame("", index=close.index, columns=close.columns, dtype="object")
    signal_type = signal_type.mask(reclaim_candidate, "RECLAIM")
    signal_type = signal_type.mask(leader_candidate, "LEADER")
    signal_type = signal_type.mask(momentum_candidate, "MOMENTUM")
    signal_type = signal_type.mask(breakout_candidate, "BREAKOUT")
    signal_type = signal_type.mask(setup_candidate, "SETUP")

    base_score = (
        0.28 * setup_readiness
        + 0.20 * tightness
        + 0.18 * trend_strength
        + 0.16 * ma_stretch_score
        + 0.12 * breakout_extension_score
        + 0.06 * freshness_score
        + 0.06 * signal_type_bonus
    )

    reclaim_trend_strength = ((ma50 / ma150) - 0.97).clip(lower=0, upper=0.08) / 0.08
    reclaim_rs_strength = (
        ((rs_vs_spy_20d - scanner.RECLAIM_MIN_RS_VS_SPY_20D).clip(lower=0, upper=10) / 10)
        + ((rs_vs_qqq_20d - scanner.RECLAIM_MIN_RS_VS_QQQ_20D).clip(lower=0, upper=10) / 10)
    ) / 2
    reclaim_extension_score = 1 - ((extension_from_ma20 - 0.02).abs() / (scanner.RECLAIM_MAX_ABOVE_MA20_PCT / 100)).clip(lower=0, upper=1)
    reclaim_trigger_score = ((close / close.rolling(10).max().shift(1)) - 1).clip(lower=0, upper=0.03) / 0.03
    reclaim_score = 0.30 * reclaim_trend_strength + 0.30 * reclaim_rs_strength + 0.25 * reclaim_extension_score + 0.15 * reclaim_trigger_score
    reclaim_readiness = (
        0.25 * scanner.soft_clip_ratio(ma50 / ma150, 0.95, 1.01)
        + 0.20 * scanner.soft_clip_ratio(close / ma20, 0.99, 1.02)
        + 0.15 * scanner.soft_clip_ratio(ma20 / ma20.shift(3), 0.995, 1.01)
        + 0.20 * scanner.soft_clip_ratio(rs_vs_spy_20d, -2.0, 5.0)
        + 0.20 * scanner.soft_clip_ratio(rs_vs_qqq_20d, -3.0, 3.0)
    )

    momentum_rs_strength = (
        ((rs_vs_spy_20d - scanner.MOMENTUM_MIN_RS_VS_SPY_20D).clip(lower=0, upper=30) / 30)
        + ((rs_vs_qqq_20d - scanner.MOMENTUM_MIN_RS_VS_QQQ_20D).clip(lower=0, upper=25) / 25)
    ) / 2
    momentum_extension_score = 1 - ((extension_from_ma20 - 0.10).abs() / (scanner.MOMENTUM_MAX_ABOVE_MA20_PCT / 100)).clip(lower=0, upper=1)
    momentum_score = 0.28 * trend_strength + 0.32 * momentum_rs_strength + 0.20 * momentum_extension_score + 0.10 * breakout_extension_score + 0.10 * freshness_score
    momentum_readiness = (
        0.25 * scanner.soft_clip_ratio(rs_vs_spy_20d, 5.0, 20.0)
        + 0.20 * scanner.soft_clip_ratio(rs_vs_qqq_20d, 0.0, 15.0)
        + 0.20 * scanner.soft_clip_ratio(scanner.MAX_BREAKOUT_AGE - breakout_age, -2.0, 2.0)
        + 0.15 * scanner.soft_clip_ratio(scanner.MOMENTUM_MAX_10D_RUNUP_PCT - runup_10d_pct, -10.0, 8.0)
        + 0.20 * (1 - ((extension_from_ma20 - 0.10).abs() / 0.15).clip(lower=0, upper=1))
    )

    leader_rs_strength = (
        ((rs_vs_spy_20d - scanner.LEADER_MIN_RS_VS_SPY_20D).clip(lower=0, upper=20) / 20)
        + ((rs_vs_qqq_20d - scanner.LEADER_MIN_RS_VS_QQQ_20D).clip(lower=0, upper=15) / 15)
    ) / 2
    leader_extension_score = 1 - ((extension_from_ma20 - 0.09).abs() / 0.10).clip(lower=0, upper=1)
    leader_proximity_score = 1 - (((-extension_from_breakout).clip(lower=0)) / 0.30).clip(lower=0, upper=1)
    leader_persistence_score = 1 - (breakout_age / scanner.LEADER_MAX_BREAKOUT_AGE).clip(lower=0, upper=1)
    leader_score = 0.18 * trend_strength + 0.40 * leader_rs_strength + 0.18 * leader_extension_score + 0.10 * leader_proximity_score + 0.08 * leader_persistence_score + 0.12
    leader_readiness = (
        0.18 * scanner.soft_clip_ratio(ma50 / ma150, 0.98, 1.06)
        + 0.30 * scanner.soft_clip_ratio(rs_vs_spy_20d, 2.0, 18.0)
        + 0.20 * scanner.soft_clip_ratio(rs_vs_qqq_20d, 0.0, 12.0)
        + 0.17 * (1 - ((extension_from_ma20 - 0.09).abs() / 0.14).clip(lower=0, upper=1))
        + 0.08 * (1 - (((-extension_from_breakout).clip(lower=0)) / 0.35).clip(lower=0, upper=1))
        + 0.07 * scanner.soft_clip_ratio(scanner.LEADER_MAX_BREAKOUT_AGE - breakout_age, 0.0, 25.0)
    )

    raw_score = pd.concat(
        [
            base_score.stack(dropna=False).rename("base"),
            (reclaim_score * reclaim_readiness).stack(dropna=False).rename("reclaim"),
            (momentum_score * momentum_readiness).stack(dropna=False).rename("momentum"),
            (leader_score * leader_readiness).stack(dropna=False).rename("leader"),
        ],
        axis=1,
    ).max(axis=1).unstack()
    raw_score = raw_score.mul(macro_state["ScannerMultiplier"], axis=0)
    score = raw_score.where(candidate)
    daily_rank = score.rank(axis=1, ascending=False, method="first")

    stock_change_proxy = rs_vs_spy_20d.add(macro_state["SPY20dPct"], axis=0)
    return {
        "close": close,
        "ma20": ma20,
        "ma50": ma50,
        "ma150": ma150,
        "runup_10d_pct": runup_10d_pct,
        "avg_dollar_vol_20d": avg_dollar_vol_20d,
        "trend": trend,
        "breakout_age": breakout_age,
        "extension_from_ma20": extension_from_ma20,
        "extension_from_breakout": extension_from_breakout,
        "stock_change_20d_pct": stock_change_proxy,
        "candidate": candidate,
        "signal_type": signal_type,
        "raw_score": raw_score,
        "score": score,
        "daily_rank": daily_rank,
        "rs_vs_spy_20d_custom": rs_vs_spy_20d,
        "rs_vs_qqq_20d_custom": rs_vs_qqq_20d,
    }


def simulate_with_custom_rs(
    state: dict[str, pd.DataFrame],
    exit_state: dict[str, pd.DataFrame],
    leveraged_flags: pd.Series,
    macro_state: pd.DataFrame,
    min_rs_vs_spy_20d: float,
    min_rs_vs_qqq_20d: float,
) -> pd.DataFrame:
    candidate = state["candidate"]
    signal_type_state = state["signal_type"]
    score = state["score"]
    daily_rank = state["daily_rank"]
    close = state["close"]
    rs_spy = state["rs_vs_spy_20d_custom"]
    rs_qqq = state["rs_vs_qqq_20d_custom"]

    dates = candidate.index
    analysis_dates = dates[-system_signal_backtest.DEFAULT_ANALYSIS_DAYS:] if len(dates) > system_signal_backtest.DEFAULT_ANALYSIS_DAYS else dates

    open_positions: dict[str, dict[str, object]] = {}
    trades: list[dict[str, object]] = []

    for current_date in analysis_dates:
        todays_entries = score.loc[current_date].dropna().sort_values(ascending=False).index.tolist()
        entries_opened_today = 0
        for symbol in todays_entries:
            if symbol in open_positions:
                continue
            entry_price = close.at[current_date, symbol]
            if pd.isna(entry_price):
                continue
            signal_type = str(signal_type_state.at[current_date, symbol])
            trade_action = scanner.build_trade_action(signal_type, 0.0 if signal_type == "BREAKOUT" else np.nan, 0.0)
            if trade_action != "BUY":
                continue
            entry_rank = daily_rank.at[current_date, symbol]
            entry_score = score.at[current_date, symbol]
            if pd.isna(entry_score) or entry_score < 0.60:
                continue
            if pd.notna(entry_rank) and int(entry_rank) > 10:
                continue
            if macro_state.at[current_date, "MacroRegime"] == "Risk-off":
                continue
            rs_vs_spy_20d = rs_spy.at[current_date, symbol]
            rs_vs_qqq_20d = rs_qqq.at[current_date, symbol]
            if pd.notna(rs_vs_spy_20d) and rs_vs_spy_20d < min_rs_vs_spy_20d:
                continue
            if pd.notna(rs_vs_qqq_20d) and rs_vs_qqq_20d < min_rs_vs_qqq_20d:
                continue
            if entries_opened_today >= 4:
                break
            open_positions[symbol] = {
                "entry_date": current_date,
                "entry_price": float(entry_price),
                "entry_signal_type": signal_type,
                "entry_rank": int(entry_rank) if pd.notna(entry_rank) else np.nan,
                "entry_score": round(float(entry_score), 3),
                "pending_exit_days": 0,
                "peak_pnl_pct": 0.0,
            }
            entries_opened_today += 1

        exit_symbols: list[str] = []
        for symbol, position in open_positions.items():
            if current_date <= position["entry_date"]:
                continue
            last_price = exit_state["close"].at[current_date, symbol]
            if pd.isna(last_price):
                continue
            pnl_pct = ((float(last_price) / float(position["entry_price"])) - 1) * 100
            hold_days = int((current_date - position["entry_date"]).days)
            position["peak_pnl_pct"] = max(float(position.get("peak_pnl_pct", pnl_pct)), float(pnl_pct))

            review_row = pd.Series(
                {
                    "Side": "LONG",
                    "LastPrice": float(last_price),
                    "MA20": exit_state["ma20"].at[current_date, symbol],
                    "MA50": exit_state["ma50"].at[current_date, symbol],
                    "MA150": exit_state["ma150"].at[current_date, symbol],
                    "PnLPct": pnl_pct,
                    "Runup10dPct": exit_state["runup_10d_pct"].at[current_date, symbol],
                    "AvgDollarVol20dM": (
                        exit_state["avg_dollar_vol_20d"].at[current_date, symbol] / 1_000_000
                        if pd.notna(exit_state["avg_dollar_vol_20d"].at[current_date, symbol])
                        else np.nan
                    ),
                    "MACDHist": exit_state["macd_hist"].at[current_date, symbol],
                    "MA20Slope": exit_state["ma20_slope"].at[current_date, symbol],
                    "BreakoutAge": state["breakout_age"].at[current_date, symbol],
                    "DistFromMA20Pct": state["extension_from_ma20"].at[current_date, symbol] * 100,
                    "DistFromBreakoutPct": state["extension_from_breakout"].at[current_date, symbol] * 100,
                    "ScannerScore": state["score"].at[current_date, symbol],
                    "ScannerTrend": state["trend"].at[current_date, symbol],
                    "ScannerSignalType": str(state["signal_type"].at[current_date, symbol]),
                }
            )
            macro_snapshot = {
                "regime": macro_state.at[current_date, "MacroRegime"],
                "score": macro_state.at[current_date, "MacroScore"],
                "scanner_multiplier": macro_state.at[current_date, "ScannerMultiplier"],
                "exit_review_threshold_shift": macro_state.at[current_date, "ExitReviewThresholdShift"],
            }
            recommendation, reason = system_signal_backtest.build_recommendation(
                review_row,
                bool(leveraged_flags.get(symbol, False)),
                macro_snapshot,
            )
            exit_signal = recommendation == "EXIT"
            if exit_signal:
                position["pending_exit_days"] = int(position.get("pending_exit_days", 0)) + 1
            else:
                position["pending_exit_days"] = 0
            if position["pending_exit_days"] < 3:
                continue
            exit_symbols.append(symbol)
            trades.append(
                {
                    "Symbol": symbol,
                    "EntryDate": position["entry_date"].date(),
                    "ExitDate": current_date.date(),
                    "EntrySignalType": position["entry_signal_type"],
                    "EntryRank": position["entry_rank"],
                    "EntryScore": position["entry_score"],
                    "EntryPrice": round(float(position["entry_price"]), 4),
                    "ExitPrice": round(float(last_price), 4),
                    "PnLPct": round(float(pnl_pct), 2),
                    "HoldDays": hold_days,
                    "ExitRecommendation": recommendation,
                    "ExitReason": reason,
                }
            )
        for symbol in exit_symbols:
            open_positions.pop(symbol, None)
    return pd.DataFrame(trades)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    universe = system_signal_backtest.load_universe(str(repo_root / "dataset" / "combined_universe.csv"))
    leveraged_flags = system_signal_backtest.build_local_leveraged_flags(universe)
    symbols = universe["Symbol"].tolist()
    close, high, low, volume = system_signal_backtest.download_universe_history(symbols)
    usable_symbols = close.columns.tolist()
    leveraged_flags = leveraged_flags.reindex(usable_symbols).fillna(False)
    macro_state = system_signal_backtest.compute_macro_state(close.index)
    macro_close = pd.DataFrame(
        {
            "SPY": close.index.to_series().map({idx: np.nan for idx in close.index}),
            "QQQ": close.index.to_series().map({idx: np.nan for idx in close.index}),
        },
        index=close.index,
    )
    # Pull benchmark closes directly to avoid approximating from percent changes.
    raw = system_signal_backtest.yf.download(
        tickers=["SPY", "QQQ"],
        period=system_signal_backtest.DOWNLOAD_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    macro_close = pd.DataFrame(
        {
            "SPY": system_signal_backtest.extract_field_frame(raw, "Close", ["SPY"])["SPY"],
            "QQQ": system_signal_backtest.extract_field_frame(raw, "Close", ["QQQ"])["QQQ"],
        }
    ).reindex(close.index).ffill()

    exit_state = system_signal_backtest.compute_exit_state(close, volume)
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        rs_spy, rs_qqq = compute_rs_frames(close, macro_close, variant.name)
        state = compute_state_with_rs(close, high, low, volume, leveraged_flags, macro_state, "ma20", rs_spy, rs_qqq)
        for rs_spy_min, rs_qqq_min in THRESHOLD_GRID:
            trades_df = simulate_with_custom_rs(state, exit_state, leveraged_flags, macro_state, rs_spy_min, rs_qqq_min)
            closed = trades_df[trades_df["ExitDate"].notna()].copy() if not trades_df.empty else trades_df
            closed_trades = len(closed)
            avg_ret = float(closed["PnLPct"].mean()) if closed_trades else np.nan
            median_ret = float(closed["PnLPct"].median()) if closed_trades else np.nan
            win_rate = float((closed["PnLPct"] > 0).mean() * 100) if closed_trades else np.nan
            avg_hold = float(closed["HoldDays"].mean()) if closed_trades else np.nan
            annualized = (
                (((1 + (avg_ret / 100.0)) ** (252.0 / avg_hold)) - 1) * 100.0
                if closed_trades and avg_hold > 0 and avg_ret > -100
                else np.nan
            )
            rows.append(
                {
                    "variant": variant.name,
                    "description": variant.description,
                    "rs_spy_min": rs_spy_min,
                    "rs_qqq_min": rs_qqq_min,
                    "closed_trades": closed_trades,
                    "win_rate_pct": round(win_rate, 2) if pd.notna(win_rate) else np.nan,
                    "avg_ret_pct": round(avg_ret, 2) if pd.notna(avg_ret) else np.nan,
                    "median_ret_pct": round(median_ret, 2) if pd.notna(median_ret) else np.nan,
                    "avg_hold_days": round(avg_hold, 1) if pd.notna(avg_hold) else np.nan,
                    "annualized_trade_pct": round(annualized, 2) if pd.notna(annualized) else np.nan,
                }
            )
    results = pd.DataFrame(rows)
    results = results.sort_values(["avg_ret_pct", "closed_trades"], ascending=[False, False]).reset_index(drop=True)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
