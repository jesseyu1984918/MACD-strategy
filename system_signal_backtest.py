from __future__ import annotations

import argparse
from pathlib import Path
import sys

VENV_SITE_PACKAGES = Path(__file__).resolve().parent / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import cache as yf_cache

import macro_market_status
import scanner
from position_exit_review import build_recommendation
from trade_guardrails import LEVERAGED_HINTS, MIN_AVG_DOLLAR_VOL, MIN_PRICE


DOWNLOAD_PERIOD = "2y"
CHUNK_SIZE = 50
DEFAULT_ANALYSIS_DAYS = 252
TRADES_OUT = "system_backtest_trades.csv"
SUMMARY_OUT = "system_backtest_summary.txt"
DAILY_SIGNALS_OUT = "system_backtest_daily_signals.csv"
YF_CACHE_DIR = Path(__file__).resolve().parent / ".yf_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest buy/sell dates from the current scanner and exit-review system."
    )
    parser.add_argument("--universe", default="combined_universe.csv")
    parser.add_argument("--analysis-days", type=int, default=DEFAULT_ANALYSIS_DAYS)
    parser.add_argument("--trend-mode", choices=list(scanner.TREND_MODES), default="ma50")
    parser.add_argument(
        "--allowed-entry-signals",
        default="ALL",
        help="Comma-separated signal types to allow for entries, e.g. LEADER,MOMENTUM. Default ALL.",
    )
    parser.add_argument(
        "--entry-mode",
        choices=["all_candidates", "breakout_only"],
        default="breakout_only",
    )
    parser.add_argument(
        "--exit-on",
        choices=["review_or_exit", "exit_only"],
        default="review_or_exit",
    )
    parser.add_argument("--min-entry-score", type=float, default=0.0)
    parser.add_argument(
        "--leader-min-entry-score",
        type=float,
        default=-1.0,
        help="Optional stricter minimum score for LEADER entries. Negative means use --min-entry-score.",
    )
    parser.add_argument("--max-entry-rank", type=int, default=0)
    parser.add_argument("--max-new-positions-per-day", type=int, default=0)
    parser.add_argument("--exit-confirm-days", type=int, default=1)
    parser.add_argument("--early-failure-days", type=int, default=0)
    parser.add_argument("--early-failure-loss-pct", type=float, default=0.0)
    parser.add_argument("--profit-lock-trigger-pct", type=float, default=0.0)
    parser.add_argument("--profit-lock-drawdown-pct", type=float, default=0.0)
    parser.add_argument("--min-rs-vs-spy-20d", type=float, default=-999.0)
    parser.add_argument("--min-rs-vs-qqq-20d", type=float, default=-999.0)
    parser.add_argument(
        "--allow-risk-off-entries",
        action="store_true",
        help="If omitted, new entries are skipped when the macro regime is Risk-off.",
    )
    parser.add_argument("--trades-out", default=TRADES_OUT)
    parser.add_argument("--summary-out", default=SUMMARY_OUT)
    parser.add_argument("--daily-signals-out", default=DAILY_SIGNALS_OUT)
    return parser.parse_args()


def load_universe(csv_path: str) -> pd.DataFrame:
    universe = pd.read_csv(csv_path)
    universe.columns = [str(col).strip() for col in universe.columns]
    if "Symbol" not in universe.columns:
        raise ValueError(f"Universe file {csv_path} must include a Symbol column")
    if "Description" not in universe.columns:
        universe["Description"] = ""
    universe["Symbol"] = universe["Symbol"].astype(str).str.strip().str.upper()
    universe["Description"] = universe["Description"].fillna("").astype(str)
    universe = universe[universe["Symbol"].ne("")].drop_duplicates("Symbol").reset_index(drop=True)
    return universe


def build_local_leveraged_flags(universe: pd.DataFrame) -> pd.Series:
    flags: dict[str, bool] = {}
    for row in universe.itertuples(index=False):
        text = f"{row.Symbol} {row.Description}".lower()
        is_etf = "etf" in text or "trust" in text or "fund" in text
        leveraged = any(token in text for token in LEVERAGED_HINTS)
        inverse = "inverse" in text or "short" in text
        # Match the live scanner: ETF universes are allowed to include
        # leveraged and inverse products without being auto-blocked.
        flags[row.Symbol] = (leveraged or inverse) and not is_etf
    return pd.Series(flags, dtype=bool)


def extract_field_frame(raw: pd.DataFrame, field: str, expected_columns: list[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = list(raw.columns.get_level_values(0))
        level1 = list(raw.columns.get_level_values(1))
        if field in level0:
            frame = raw[field].copy()
        elif field in level1:
            frame = raw.xs(field, axis=1, level=1).copy()
        else:
            return pd.DataFrame(index=raw.index, columns=expected_columns, dtype=float)
    else:
        if field not in raw.columns:
            return pd.DataFrame(index=raw.index, columns=expected_columns, dtype=float)
        frame = raw[[field]].copy()
        frame.columns = expected_columns[:1]

    if isinstance(frame, pd.Series):
        frame = frame.to_frame()

    frame.columns = [str(col).strip().upper() for col in frame.columns]
    frame = frame.reindex(columns=expected_columns)
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.astype(float)


def download_universe_history(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close_frames: list[pd.DataFrame] = []
    high_frames: list[pd.DataFrame] = []
    low_frames: list[pd.DataFrame] = []
    volume_frames: list[pd.DataFrame] = []

    for start in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[start : start + CHUNK_SIZE]
        print(f"Downloading {start + 1}-{start + len(chunk)} of {len(symbols)}")
        try:
            raw = yf.download(
                tickers=chunk,
                period=DOWNLOAD_PERIOD,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )
        except Exception as exc:
            print(f"Chunk download failed for {chunk[0]}..{chunk[-1]}: {exc}")
            continue

        if raw is None or raw.empty:
            continue

        expected_columns = [symbol.upper() for symbol in chunk]
        close_frames.append(extract_field_frame(raw, "Close", expected_columns))
        high_frames.append(extract_field_frame(raw, "High", expected_columns))
        low_frames.append(extract_field_frame(raw, "Low", expected_columns))
        volume_frames.append(extract_field_frame(raw, "Volume", expected_columns))

    if not close_frames:
        raise SystemExit("No market data downloaded for the universe")

    close = pd.concat(close_frames, axis=1).sort_index()
    high = pd.concat(high_frames, axis=1).reindex(close.index)
    low = pd.concat(low_frames, axis=1).reindex(close.index)
    volume = pd.concat(volume_frames, axis=1).reindex(close.index)

    close = close.loc[:, ~close.columns.duplicated()]
    high = high.loc[:, ~high.columns.duplicated()]
    low = low.loc[:, ~low.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]

    valid_symbols = [
        symbol
        for symbol in close.columns
        if close[symbol].dropna().shape[0] >= scanner.MIN_HISTORY_BARS
    ]
    close = close[valid_symbols].ffill()
    high = high[valid_symbols].ffill()
    low = low[valid_symbols].ffill()
    volume = volume[valid_symbols].ffill()
    return close, high, low, volume


def compute_macro_state(index: pd.Index) -> pd.DataFrame:
    macro_symbols = list(macro_market_status.MACRO_SYMBOLS.keys())
    raw = yf.download(
        tickers=macro_symbols,
        period=DOWNLOAD_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(
            {
                "MacroRegime": "Mixed",
                "MacroTrend": "No data",
                "MacroRisk": "No data",
                "MacroScore": 0,
                "ScannerMultiplier": 1.0,
                "ExitReviewThresholdShift": 0.0,
                "SPY20dPct": np.nan,
                "QQQ20dPct": np.nan,
            },
            index=index,
        )

    close = extract_field_frame(raw, "Close", macro_symbols).reindex(index).ffill()
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    change_20d_pct = close.pct_change(20) * 100

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal

    macro_rows: list[dict[str, object]] = []
    for current_date in index:
        statuses: dict[str, str] = {}
        for symbol in macro_symbols:
            row_close = close.at[current_date, symbol] if symbol in close.columns else np.nan
            row_ma50 = ma50.at[current_date, symbol] if symbol in ma50.columns else np.nan
            row_ma150 = ma150.at[current_date, symbol] if symbol in ma150.columns else np.nan
            row_macd = macd_hist.at[current_date, symbol] if symbol in macd_hist.columns else np.nan

            if pd.isna(row_ma50) or pd.isna(row_macd) or pd.isna(row_close):
                statuses[symbol] = "Insufficient history"
            elif symbol == "^VIX":
                statuses[symbol] = "Calm volatility backdrop" if row_close < row_ma50 else "Elevated volatility backdrop"
            else:
                above_ma50 = row_close > row_ma50
                above_ma150 = row_close > row_ma150 if pd.notna(row_ma150) else False
                macd_positive = row_macd > 0
                if above_ma50 and above_ma150 and macd_positive:
                    statuses[symbol] = "Bullish trend"
                elif above_ma50 and above_ma150:
                    statuses[symbol] = "Uptrend, momentum soft"
                elif above_ma50 or macd_positive:
                    statuses[symbol] = "Mixed"
                else:
                    statuses[symbol] = "Weak"

        bullish_count = sum(statuses.get(symbol) == "Bullish trend" for symbol in ["SPY", "QQQ", "IWM", "DIA"])
        weak_count = sum(statuses.get(symbol) == "Weak" for symbol in ["SPY", "QQQ", "IWM", "DIA"])
        if bullish_count >= 3:
            trend = "Broad equity trend is supportive"
        elif weak_count >= 2:
            trend = "Broad equity trend is fragile"
        else:
            trend = "Broad equity trend is mixed"

        risk = "Neutral"
        if statuses.get("^VIX") == "Elevated volatility backdrop":
            risk = "Risk-off pressure from volatility"

        hyg_20d = change_20d_pct.at[current_date, "HYG"] if "HYG" in change_20d_pct.columns else np.nan
        tlt_20d = change_20d_pct.at[current_date, "TLT"] if "TLT" in change_20d_pct.columns else np.nan
        if pd.notna(hyg_20d) and pd.notna(tlt_20d):
            if hyg_20d > tlt_20d and risk == "Neutral":
                risk = "Risk appetite is leaning constructive"
            elif tlt_20d > hyg_20d:
                risk = "Defensive assets are leading"

        if "supportive" in trend.lower() and "constructive" in risk.lower():
            regime = "Risk-on"
            score = 1
            scanner_multiplier = 1.05
            exit_review_threshold_shift = -0.03
        elif "fragile" in trend.lower() or "defensive" in risk.lower() or "risk-off" in risk.lower():
            regime = "Risk-off"
            score = -1
            scanner_multiplier = 0.88
            exit_review_threshold_shift = 0.05
        else:
            regime = "Mixed"
            score = 0
            scanner_multiplier = 1.0
            exit_review_threshold_shift = 0.0

        macro_rows.append(
            {
                "MacroRegime": regime,
                "MacroTrend": trend,
                "MacroRisk": risk,
                "MacroScore": score,
                "ScannerMultiplier": scanner_multiplier,
                "ExitReviewThresholdShift": exit_review_threshold_shift,
                "SPY20dPct": change_20d_pct.at[current_date, "SPY"] if "SPY" in change_20d_pct.columns else np.nan,
                "QQQ20dPct": change_20d_pct.at[current_date, "QQQ"] if "QQQ" in change_20d_pct.columns else np.nan,
            }
        )

    return pd.DataFrame(macro_rows, index=index)


def compute_scanner_state(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    leveraged_flags: pd.Series,
    macro_state: pd.DataFrame,
    trend_mode: str = "ma50",
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
    price_ok = close >= MIN_PRICE
    liquidity_ok = avg_dollar_vol_20d >= MIN_AVG_DOLLAR_VOL
    runup_ok = runup_10d_pct <= scanner.MAX_10D_RUNUP_PCT
    not_extended = extension_from_ma20 <= (scanner.MAX_ABOVE_MA20_PCT / 100)
    setup_ma50_not_extended = scanner.build_ma50_extension_mask(
        extension_from_ma50,
        trend_mode,
        scanner.SETUP_MAX_ABOVE_MA50_PCT,
    )
    breakout_ma50_not_extended = scanner.build_ma50_extension_mask(
        extension_from_ma50,
        trend_mode,
        scanner.BREAKOUT_MAX_ABOVE_MA50_PCT,
    )
    momentum_ma50_not_extended = scanner.build_ma50_extension_mask(
        extension_from_ma50,
        trend_mode,
        scanner.MOMENTUM_MAX_ABOVE_MA50_PCT,
    )
    leader_ma50_not_extended = scanner.build_ma50_extension_mask(
        extension_from_ma50,
        trend_mode,
        scanner.LEADER_MAX_ABOVE_MA50_PCT,
    )
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
    breakout_extension_score = 1 - (
        extension_from_breakout.clip(lower=0) / scanner.MAX_BREAKOUT_EXTENSION
    ).clip(lower=0, upper=1)
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
    rs_vs_spy_20d = stock_change_20d_pct.sub(macro_state["SPY20dPct"], axis=0)
    rs_vs_qqq_20d = stock_change_20d_pct.sub(macro_state["QQQ20dPct"], axis=0)

    reclaim_trend_strength = ((ma50 / ma150) - 0.97).clip(lower=0, upper=0.08) / 0.08
    reclaim_rs_strength = (
        ((rs_vs_spy_20d - scanner.RECLAIM_MIN_RS_VS_SPY_20D).clip(lower=0, upper=10) / 10)
        + ((rs_vs_qqq_20d - scanner.RECLAIM_MIN_RS_VS_QQQ_20D).clip(lower=0, upper=10) / 10)
    ) / 2
    reclaim_extension_score = 1 - (
        (extension_from_ma20 - 0.02).abs() / (scanner.RECLAIM_MAX_ABOVE_MA20_PCT / 100)
    ).clip(lower=0, upper=1)
    reclaim_trigger_score = ((close / close.rolling(10).max().shift(1)) - 1).clip(lower=0, upper=0.03) / 0.03
    reclaim_score = (
        0.30 * reclaim_trend_strength
        + 0.30 * reclaim_rs_strength
        + 0.25 * reclaim_extension_score
        + 0.15 * reclaim_trigger_score
    )
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
    momentum_extension_score = 1 - (
        (extension_from_ma20 - 0.10).abs() / (scanner.MOMENTUM_MAX_ABOVE_MA20_PCT / 100)
    ).clip(lower=0, upper=1)
    momentum_score = (
        0.28 * trend_strength
        + 0.32 * momentum_rs_strength
        + 0.20 * momentum_extension_score
        + 0.10 * breakout_extension_score
        + 0.10 * freshness_score
    )
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
    leader_extension_score = 1 - (
        (extension_from_ma20 - 0.09).abs() / 0.10
    ).clip(lower=0, upper=1)
    leader_proximity_score = 1 - (((-extension_from_breakout).clip(lower=0)) / 0.30).clip(lower=0, upper=1)
    leader_persistence_score = 1 - (breakout_age / scanner.LEADER_MAX_BREAKOUT_AGE).clip(lower=0, upper=1)
    leader_score = (
        0.18 * trend_strength
        + 0.40 * leader_rs_strength
        + 0.18 * leader_extension_score
        + 0.10 * leader_proximity_score
        + 0.08 * leader_persistence_score
        + 0.12
    )
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
    macro_signal_adjustment = pd.DataFrame(1.0, index=signal_type.index, columns=signal_type.columns)
    for current_date in signal_type.index:
        macro_signal_adjustment.loc[current_date] = scanner.build_macro_signal_adjustment(
            signal_type.loc[[current_date]],
            str(macro_state.at[current_date, "MacroRegime"]),
            str(macro_state.at[current_date, "MacroTrend"]),
            str(macro_state.at[current_date, "MacroRisk"]),
        ).iloc[0]
    raw_score = raw_score * macro_signal_adjustment
    score = raw_score.where(candidate)
    score = score.where(candidate)

    daily_rank = score.rank(axis=1, ascending=False, method="first")

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
        "stock_change_20d_pct": stock_change_20d_pct,
        "setup_candidate": setup_candidate,
        "breakout_candidate": breakout_candidate,
        "momentum_candidate": momentum_candidate,
        "leader_candidate": leader_candidate,
        "reclaim_candidate": reclaim_candidate,
        "candidate": candidate,
        "signal_type": signal_type,
        "raw_score": raw_score,
        "score": score,
        "daily_rank": daily_rank,
    }


def compute_exit_state(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    runup_10d_pct = close.pct_change(10) * 100
    avg_dollar_vol_20d = (close * volume).rolling(20).mean()
    ma20_slope = ma20.diff()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal

    return {
        "close": close,
        "ma20": ma20,
        "ma50": ma50,
        "ma150": ma150,
        "runup_10d_pct": runup_10d_pct,
        "avg_dollar_vol_20d": avg_dollar_vol_20d,
        "ma20_slope": ma20_slope,
        "macd_hist": macd_hist,
    }


def simulate_trades(
    state: dict[str, pd.DataFrame],
    exit_state: dict[str, pd.DataFrame],
    leveraged_flags: pd.Series,
    macro_state: pd.DataFrame,
    analysis_days: int,
    entry_mode: str,
    exit_on: str,
    min_entry_score: float,
    leader_min_entry_score: float,
    max_entry_rank: int,
    max_new_positions_per_day: int,
    exit_confirm_days: int,
    early_failure_days: int,
    early_failure_loss_pct: float,
    profit_lock_trigger_pct: float,
    profit_lock_drawdown_pct: float,
    min_rs_vs_spy_20d: float,
    min_rs_vs_qqq_20d: float,
    allow_risk_off_entries: bool,
    allowed_entry_signals: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = state["candidate"]
    signal_type_state = state["signal_type"]
    score = state["score"]
    daily_rank = state["daily_rank"]
    close = state["close"]

    dates = candidate.index
    analysis_dates = dates[-analysis_days:] if len(dates) > analysis_days else dates

    open_positions: dict[str, dict[str, object]] = {}
    trades: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []

    for current_date in analysis_dates:
        todays_entries = (
            score.loc[current_date]
            .dropna()
            .sort_values(ascending=False)
            .index.tolist()
        )

        entries_opened_today = 0
        for symbol in todays_entries:
            if symbol in open_positions:
                continue

            entry_price = close.at[current_date, symbol]
            if pd.isna(entry_price):
                continue

            signal_type = str(signal_type_state.at[current_date, symbol])
            if allowed_entry_signals and signal_type not in allowed_entry_signals:
                continue
            trade_action = scanner.build_trade_action(signal_type, 0.0 if signal_type == "BREAKOUT" else np.nan, 0.0)
            if entry_mode == "breakout_only" and trade_action != "BUY":
                continue

            entry_rank = daily_rank.at[current_date, symbol]
            entry_score = score.at[current_date, symbol]
            macro_regime = macro_state.at[current_date, "MacroRegime"]
            if pd.isna(entry_score):
                continue
            required_entry_score = leader_min_entry_score if signal_type == "LEADER" and leader_min_entry_score >= 0 else min_entry_score
            if entry_score < required_entry_score:
                continue
            if max_entry_rank > 0 and pd.notna(entry_rank) and int(entry_rank) > max_entry_rank:
                continue
            if not allow_risk_off_entries and macro_regime == "Risk-off":
                continue
            stock_20d = state["stock_change_20d_pct"].at[current_date, symbol]
            spy_20d = macro_state.at[current_date, "SPY20dPct"]
            qqq_20d = macro_state.at[current_date, "QQQ20dPct"]
            rs_vs_spy_20d = stock_20d - spy_20d if pd.notna(stock_20d) and pd.notna(spy_20d) else np.nan
            rs_vs_qqq_20d = stock_20d - qqq_20d if pd.notna(stock_20d) and pd.notna(qqq_20d) else np.nan
            if pd.notna(rs_vs_spy_20d) and rs_vs_spy_20d < min_rs_vs_spy_20d:
                continue
            if pd.notna(rs_vs_qqq_20d) and rs_vs_qqq_20d < min_rs_vs_qqq_20d:
                continue
            if max_new_positions_per_day > 0 and entries_opened_today >= max_new_positions_per_day:
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
            daily_rows.append(
                {
                    "Date": current_date.date(),
                    "Symbol": symbol,
                    "Action": "BUY",
                    "SignalType": signal_type,
                    "Rank": int(entry_rank) if pd.notna(entry_rank) else np.nan,
                    "Score": round(float(entry_score), 3),
                    "Price": round(float(entry_price), 4),
                    "Recommendation": None,
                    "Reason": "scanner_candidate",
                    "MacroRegime": macro_regime,
                    "RSvsSPY20d": round(float(rs_vs_spy_20d), 2) if pd.notna(rs_vs_spy_20d) else np.nan,
                    "RSvsQQQ20d": round(float(rs_vs_qqq_20d), 2) if pd.notna(rs_vs_qqq_20d) else np.nan,
                }
            )

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

            recommendation, reason = build_recommendation(
                review_row,
                bool(leveraged_flags.get(symbol, False)),
                macro_snapshot,
            )
            if (
                early_failure_days > 0
                and hold_days <= early_failure_days
                and pnl_pct <= -abs(early_failure_loss_pct)
            ):
                recommendation = "EXIT"
                reason = (
                    f"early_failure_stop_{early_failure_days}d_{abs(early_failure_loss_pct):.1f}pct; "
                    + str(reason)
                )
            peak_pnl_pct = float(position.get("peak_pnl_pct", pnl_pct))
            if (
                profit_lock_trigger_pct > 0
                and profit_lock_drawdown_pct > 0
                and peak_pnl_pct >= profit_lock_trigger_pct
                and pnl_pct <= peak_pnl_pct - profit_lock_drawdown_pct
            ):
                recommendation = "EXIT"
                reason = (
                    f"profit_lock_{profit_lock_trigger_pct:.1f}_{profit_lock_drawdown_pct:.1f}; "
                    + str(reason)
                )

            exit_signal = recommendation != "HOLD" if exit_on == "review_or_exit" else recommendation == "EXIT"
            if exit_signal:
                position["pending_exit_days"] = int(position.get("pending_exit_days", 0)) + 1
            else:
                position["pending_exit_days"] = 0
            should_exit = bool(position["pending_exit_days"] >= max(exit_confirm_days, 1))
            if not should_exit:
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
            daily_rows.append(
                {
                    "Date": current_date.date(),
                    "Symbol": symbol,
                    "Action": "SELL",
                    "SignalType": position["entry_signal_type"],
                    "Rank": position["entry_rank"],
                    "Score": position["entry_score"],
                    "Price": round(float(last_price), 4),
                    "Recommendation": recommendation,
                    "Reason": reason,
                    "MacroRegime": macro_snapshot["regime"],
                }
            )

        for symbol in exit_symbols:
            open_positions.pop(symbol, None)

    if analysis_dates.empty:
        return pd.DataFrame(), pd.DataFrame()

    final_date = analysis_dates[-1]
    for symbol, position in open_positions.items():
        last_price = close.at[final_date, symbol]
        if pd.isna(last_price):
            continue
        hold_days = int((final_date - position["entry_date"]).days)
        pnl_pct = ((float(last_price) / float(position["entry_price"])) - 1) * 100
        trades.append(
            {
                "Symbol": symbol,
                "EntryDate": position["entry_date"].date(),
                "ExitDate": None,
                "EntrySignalType": position["entry_signal_type"],
                "EntryRank": position["entry_rank"],
                "EntryScore": position["entry_score"],
                "EntryPrice": round(float(position["entry_price"]), 4),
                "ExitPrice": round(float(last_price), 4),
                "PnLPct": round(float(pnl_pct), 2),
                "HoldDays": hold_days,
                "ExitRecommendation": "OPEN",
                "ExitReason": "still_open_at_end_of_backtest",
            }
        )

    trades_df = pd.DataFrame(trades).sort_values(["EntryDate", "Symbol"]).reset_index(drop=True)
    daily_df = pd.DataFrame(daily_rows).sort_values(["Date", "Action", "Symbol"]).reset_index(drop=True)
    return trades_df, daily_df


def build_summary(
    trades_df: pd.DataFrame,
    universe_size: int,
    valid_symbols: int,
    analysis_days: int,
    trend_mode: str,
    entry_mode: str,
    exit_on: str,
    min_entry_score: float,
    leader_min_entry_score: float,
    max_entry_rank: int,
    max_new_positions_per_day: int,
    exit_confirm_days: int,
    early_failure_days: int,
    early_failure_loss_pct: float,
    profit_lock_trigger_pct: float,
    profit_lock_drawdown_pct: float,
    min_rs_vs_spy_20d: float,
    min_rs_vs_qqq_20d: float,
    allow_risk_off_entries: bool,
) -> str:
    closed = trades_df[trades_df["ExitRecommendation"].ne("OPEN")].copy()
    open_trades = trades_df[trades_df["ExitRecommendation"].eq("OPEN")].copy()

    lines = [
        "System Signal Backtest",
        f"Universe requested: {universe_size}",
        f"Universe with usable history: {valid_symbols}",
        f"Analysis window (trading days): {analysis_days}",
        f"Trend mode: {trend_mode}",
        f"Entry mode: {entry_mode}",
        f"Exit trigger: {exit_on}",
        f"Minimum entry score: {min_entry_score:.2f}",
        f"Leader minimum entry score: {'same as minimum' if leader_min_entry_score < 0 else f'{leader_min_entry_score:.2f}'}",
        f"Maximum entry rank: {'none' if max_entry_rank <= 0 else max_entry_rank}",
        f"Maximum new positions per day: {'none' if max_new_positions_per_day <= 0 else max_new_positions_per_day}",
        f"Exit confirmation days: {exit_confirm_days}",
        f"Early failure days: {early_failure_days}",
        f"Early failure loss pct: {early_failure_loss_pct:.2f}",
        f"Profit lock trigger pct: {profit_lock_trigger_pct:.2f}",
        f"Profit lock drawdown pct: {profit_lock_drawdown_pct:.2f}",
        f"Minimum RS vs SPY 20d: {min_rs_vs_spy_20d:.2f}",
        f"Minimum RS vs QQQ 20d: {min_rs_vs_qqq_20d:.2f}",
        f"Allow Risk-off entries: {allow_risk_off_entries}",
        f"Trades generated: {len(trades_df)}",
        f"Closed trades: {len(closed)}",
        f"Open trades at end: {len(open_trades)}",
    ]

    if closed.empty:
        lines.append("No closed trades were generated.")
        return "\n".join(lines)

    win_rate = (closed["PnLPct"] > 0).mean() * 100
    avg_return = closed["PnLPct"].mean()
    median_return = closed["PnLPct"].median()
    avg_hold = closed["HoldDays"].mean()
    avg_entry_rank = closed["EntryRank"].mean()

    lines.extend(
        [
            f"Win rate: {win_rate:.2f}%",
            f"Average return per closed trade: {avg_return:.2f}%",
            f"Median return per closed trade: {median_return:.2f}%",
            f"Average hold days: {avg_hold:.1f}",
            f"Average entry rank: {avg_entry_rank:.1f}",
            "",
            "Closed trades by exit recommendation:",
            closed["ExitRecommendation"].value_counts().to_string(),
            "",
            "Top 10 closed trades by return:",
            closed.sort_values("PnLPct", ascending=False)
            .head(10)[["Symbol", "EntryDate", "ExitDate", "PnLPct", "EntryRank", "ExitRecommendation"]]
            .to_string(index=False),
            "",
            "Bottom 10 closed trades by return:",
            closed.sort_values("PnLPct", ascending=True)
            .head(10)[["Symbol", "EntryDate", "ExitDate", "PnLPct", "EntryRank", "ExitRecommendation"]]
            .to_string(index=False),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    YF_CACHE_DIR.mkdir(exist_ok=True)
    yf_cache.set_cache_location(str(YF_CACHE_DIR))
    universe = load_universe(args.universe)
    leveraged_flags = build_local_leveraged_flags(universe)
    symbols = universe["Symbol"].tolist()

    close, high, low, volume = download_universe_history(symbols)
    usable_symbols = close.columns.tolist()
    leveraged_flags = leveraged_flags.reindex(usable_symbols).fillna(False)

    macro_state = compute_macro_state(close.index)
    state = compute_scanner_state(close, high, low, volume, leveraged_flags, macro_state, trend_mode=args.trend_mode)
    exit_state = compute_exit_state(close, volume)
    allowed_entry_signals = None
    if str(args.allowed_entry_signals).strip().upper() != "ALL":
        allowed_entry_signals = {
            token.strip().upper()
            for token in str(args.allowed_entry_signals).split(",")
            if token.strip()
        }

    trades_df, daily_df = simulate_trades(
        state,
        exit_state,
        leveraged_flags,
        macro_state,
        args.analysis_days,
        args.entry_mode,
        args.exit_on,
        args.min_entry_score,
        args.leader_min_entry_score,
        args.max_entry_rank,
        args.max_new_positions_per_day,
        args.exit_confirm_days,
        args.early_failure_days,
        args.early_failure_loss_pct,
        args.profit_lock_trigger_pct,
        args.profit_lock_drawdown_pct,
        args.min_rs_vs_spy_20d,
        args.min_rs_vs_qqq_20d,
        args.allow_risk_off_entries,
        allowed_entry_signals,
    )

    trades_df.to_csv(args.trades_out, index=False)
    daily_df.to_csv(args.daily_signals_out, index=False)

    summary = build_summary(
        trades_df,
        len(symbols),
        len(usable_symbols),
        args.analysis_days,
        args.trend_mode,
        args.entry_mode,
        args.exit_on,
        args.min_entry_score,
        args.leader_min_entry_score,
        args.max_entry_rank,
        args.max_new_positions_per_day,
        args.exit_confirm_days,
        args.early_failure_days,
        args.early_failure_loss_pct,
        args.profit_lock_trigger_pct,
        args.profit_lock_drawdown_pct,
        args.min_rs_vs_spy_20d,
        args.min_rs_vs_qqq_20d,
        args.allow_risk_off_entries,
    )
    Path(args.summary_out).write_text(summary)
    print(summary)
    print(f"\nSaved trades to {args.trades_out}")
    print(f"Saved daily signals to {args.daily_signals_out}")
    print(f"Saved summary to {args.summary_out}")


if __name__ == "__main__":
    main()
