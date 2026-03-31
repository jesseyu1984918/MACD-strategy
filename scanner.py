from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
import sys

VENV_SITE_PACKAGES = Path(__file__).resolve().parent / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import numpy as np
import pandas as pd
import yfinance as yf

from macro_market_status import get_benchmark_change_snapshot, get_macro_regime_snapshot
from trade_guardrails import (
    MAX_10D_RUNUP_PCT,
    MAX_ABOVE_MA20_PCT,
    MIN_AVG_DOLLAR_VOL,
    MIN_PRICE,
    build_metadata_flags,
    flatten_columns,
)


LOOKBACK = "1y"
MAX_BREAKOUT_AGE = 2
MAX_BREAKOUT_EXTENSION = 0.04
SETUP_PROXIMITY = 0.03
MIN_HISTORY_BARS = 200
MIN_RS_VS_SPY_20D = 5.0
MIN_RS_VS_QQQ_20D = 0.0
BLOCKED_OUTPUT = "scanner_blocked.csv"
RANK_HISTORY_OUTPUT = "scanner_rank_history.csv"
BLOCKED_HISTORY_OUTPUT = "scanner_blocked_history.csv"
RANK_HISTORY_COLUMNS = [
    "SnapshotDate",
    "RunTimestamp",
    "Source",
    "Rank",
    "Symbol",
    "Score",
    "SignalType",
    "TradeAction",
    "NextAction",
    "BreakoutAge",
    "DistFromMA20Pct",
    "DistFromBreakoutPct",
    "Runup10dPct",
    "AvgDollarVol20dM",
    "PreviousRank",
    "RankChange",
    "PreviousSignalType",
    "LifecycleAction",
]
LEGACY_RANK_HISTORY_COLUMNS = [
    "SnapshotDate",
    "RunTimestamp",
    "Source",
    "Rank",
    "Symbol",
    "Score",
    "SignalType",
    "BreakoutAge",
    "DistFromMA20Pct",
    "DistFromBreakoutPct",
    "Runup10dPct",
    "AvgDollarVol20dM",
    "PreviousRank",
    "RankChange",
]
MID_RANK_HISTORY_COLUMNS = [
    "SnapshotDate",
    "RunTimestamp",
    "Source",
    "Rank",
    "Symbol",
    "Score",
    "SignalType",
    "NextAction",
    "BreakoutAge",
    "DistFromMA20Pct",
    "DistFromBreakoutPct",
    "Runup10dPct",
    "AvgDollarVol20dM",
    "PreviousRank",
    "RankChange",
]


def scalar(x):
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


def bars_since_true(mask: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=mask.index, columns=mask.columns, dtype=float)

    for col in mask.columns:
        last_true_idx = None
        ages = []

        for i, is_true in enumerate(mask[col].fillna(False).to_numpy()):
            if is_true:
                last_true_idx = i
                ages.append(0.0)
            elif last_true_idx is None:
                ages.append(np.nan)
            else:
                ages.append(float(i - last_true_idx))

        result[col] = ages

    return result


def parse_breakout_age(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"", "setup", "nan"}:
        return np.nan
    if text.endswith("d"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_rank_history_df(history_path: str = RANK_HISTORY_OUTPUT) -> pd.DataFrame:
    path = Path(history_path)
    if not path.exists():
        return pd.DataFrame(columns=RANK_HISTORY_COLUMNS)

    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            if row == RANK_HISTORY_COLUMNS or row == LEGACY_RANK_HISTORY_COLUMNS or row == MID_RANK_HISTORY_COLUMNS:
                continue

            if len(row) == len(RANK_HISTORY_COLUMNS):
                record = dict(zip(RANK_HISTORY_COLUMNS, row))
            elif len(row) == len(MID_RANK_HISTORY_COLUMNS):
                record = dict(zip(MID_RANK_HISTORY_COLUMNS, row))
                record["TradeAction"] = build_trade_action(
                    record["SignalType"],
                    parse_breakout_age(record["BreakoutAge"]),
                    pd.to_numeric(record["DistFromBreakoutPct"], errors="coerce"),
                )
                record["PreviousSignalType"] = pd.NA
                record["LifecycleAction"] = build_lifecycle_action(
                    record["SignalType"],
                    pd.NA,
                    parse_breakout_age(record["BreakoutAge"]),
                    pd.to_numeric(record["DistFromBreakoutPct"], errors="coerce"),
                )
            elif len(row) == len(LEGACY_RANK_HISTORY_COLUMNS):
                record = dict(zip(LEGACY_RANK_HISTORY_COLUMNS, row))
                breakout_age = parse_breakout_age(record["BreakoutAge"])
                dist_from_breakout = pd.to_numeric(record["DistFromBreakoutPct"], errors="coerce")
                record["TradeAction"] = build_trade_action(
                    record["SignalType"],
                    breakout_age,
                    dist_from_breakout,
                )
                record["NextAction"] = build_next_action(
                    record["SignalType"],
                    breakout_age,
                    dist_from_breakout,
                )
                record["PreviousSignalType"] = pd.NA
                record["LifecycleAction"] = build_lifecycle_action(
                    record["SignalType"],
                    pd.NA,
                    breakout_age,
                    dist_from_breakout,
                )
            else:
                continue

            normalized = {col: record.get(col, pd.NA) for col in RANK_HISTORY_COLUMNS}
            rows.append(normalized)

    history_df = pd.DataFrame(rows, columns=RANK_HISTORY_COLUMNS)
    if history_df.empty:
        return history_df

    for col in ["Rank", "PreviousRank", "RankChange"]:
        history_df[col] = pd.to_numeric(history_df[col], errors="coerce").astype("Int64")
    for col in ["Score", "DistFromMA20Pct", "DistFromBreakoutPct", "Runup10dPct", "AvgDollarVol20dM"]:
        history_df[col] = pd.to_numeric(history_df[col], errors="coerce")
    return history_df


def download_symbol(symbol: str) -> pd.DataFrame | None:
    try:
        df = yf.download(symbol, period=LOOKBACK, interval="1d", auto_adjust=False, progress=False)
        df = flatten_columns(df)
        if df is None or df.empty:
            print(f"BAD  {symbol} fetch returned no rows")
            return None

        df.index = pd.to_datetime(df.index).tz_localize(None)
        required_cols = ["Close", "High", "Low", "Volume"]
        if any(col not in df.columns for col in required_cols):
            print(f"BAD  {symbol} missing columns")
            return None

        df = df[required_cols].dropna()
        if len(df) < MIN_HISTORY_BARS:
            print(f"BAD  {symbol} not enough history ({len(df)})")
            return None

        print(f"OK   {symbol} rows={len(df)}")
        return df
    except Exception as exc:
        print(f"BAD  {symbol} {exc}")
        return None


def last_true_reason(ordered_reasons: list[tuple[str, bool]]) -> str:
    hits = [label for label, condition in ordered_reasons if condition]
    return "; ".join(hits) if hits else "passed"


def load_symbols(csv_path: str) -> list[str]:
    raw = pd.read_csv(csv_path, header=None)
    first_value = str(raw.iloc[0, 0]).strip().lower() if not raw.empty else ""

    if first_value in {"symbol", "symbols", "ticker", "tickers"}:
        raw = pd.read_csv(csv_path)

    symbols = raw.iloc[:, 0].dropna().astype(str).str.strip()
    symbols = symbols[symbols.str.lower().ne("symbol")]
    return symbols.tolist()


def build_next_action(signal_type: str, breakout_age: float, dist_from_breakout_pct: float) -> str:
    if signal_type == "SETUP":
        return "Watch for breakout through the pivot before entering"

    if pd.notna(breakout_age) and breakout_age <= 0:
        return "Actionable now: starter only if the breakout is holding"

    if dist_from_breakout_pct <= 1.5:
        return "Wait for a tight hold above the breakout, then enter"

    return "Avoid chasing here; only act on a controlled pullback"


def build_trade_action(signal_type: str, breakout_age: float, dist_from_breakout_pct: float) -> str:
    if signal_type == "SETUP":
        return "WATCH"

    if pd.notna(breakout_age) and breakout_age <= 0:
        return "BUY"

    if dist_from_breakout_pct <= 1.5:
        return "WATCH"

    return "WAIT"


def build_lifecycle_action(
    signal_type: str,
    previous_signal_type: object,
    breakout_age: float,
    dist_from_breakout_pct: float,
) -> str:
    previous_signal = str(previous_signal_type) if pd.notna(previous_signal_type) else ""

    if signal_type == "SETUP":
        if previous_signal == "SETUP":
            return "Carry forward: keep on watchlist for pivot breakout"
        return "New setup: add to watchlist and set pivot alert"

    if previous_signal == "SETUP":
        return "Triggered from prior setup: review now for entry"

    if pd.notna(breakout_age) and breakout_age <= 0:
        return "New breakout today: review now for entry"

    if previous_signal == "BREAKOUT":
        if dist_from_breakout_pct <= 1.5:
            return "Still actionable: watch for tight hold above breakout"
        return "Missed initial breakout: wait for pullback, do not chase"

    return "Fresh breakout: only act if it stays tight above the pivot"


def format_ranked_candidates(
    today_score: pd.Series,
    diagnostics: pd.DataFrame,
    setup_candidate: pd.DataFrame,
    macro_snapshot: dict[str, object],
    rs_snapshot: dict[str, object],
) -> pd.DataFrame:
    if today_score.empty:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Score",
                "SignalType",
                "TradeAction",
                "NextAction",
                "BreakoutAge",
                "DistFromMA20Pct",
                "DistFromBreakoutPct",
                "Runup10dPct",
                "AvgDollarVol20dM",
            ]
        )

    last_date = diagnostics.index[-1]
    rows = []

    for symbol, val in today_score.items():
        age = diagnostics.at[last_date, (symbol, "breakout_age")]
        signal_type = "SETUP" if bool(setup_candidate.at[last_date, symbol]) else "BREAKOUT"
        dist_from_breakout_pct = round(float(diagnostics.at[last_date, (symbol, "extension_from_breakout")]) * 100, 2)
        rows.append(
            {
                "Symbol": symbol,
                "Score": round(float(val), 3),
                "SignalType": signal_type,
                "TradeAction": build_trade_action(signal_type, age, dist_from_breakout_pct),
                "NextAction": build_next_action(signal_type, age, dist_from_breakout_pct),
                "BreakoutAge": "setup" if pd.isna(age) else f"{int(age)}d",
                "DistFromMA20Pct": round(float(diagnostics.at[last_date, (symbol, "extension_from_ma20")]) * 100, 2),
                "DistFromBreakoutPct": dist_from_breakout_pct,
                "Runup10dPct": round(float(diagnostics.at[last_date, (symbol, "runup_10d_pct")]), 2),
                "AvgDollarVol20dM": round(float(diagnostics.at[last_date, (symbol, "avg_dollar_vol_20d")]) / 1_000_000, 2),
                "RSvsSPY20d": round(float(rs_snapshot["rs_vs_spy_20d"].get(symbol, np.nan)), 2),
                "RSvsQQQ20d": round(float(rs_snapshot["rs_vs_qqq_20d"].get(symbol, np.nan)), 2),
                "MacroRegime": str(macro_snapshot["regime"]),
                "MacroScore": int(macro_snapshot["score"]),
                "MacroTrend": str(macro_snapshot["trend"]),
                "MacroRisk": str(macro_snapshot["risk"]),
            }
        )

    return pd.DataFrame(rows)


def add_rank_columns(ranked_df: pd.DataFrame) -> pd.DataFrame:
    if ranked_df.empty:
        ranked_df["Rank"] = pd.Series(dtype=int)
        ranked_df["PreviousRank"] = pd.Series(dtype="Int64")
        ranked_df["RankChange"] = pd.Series(dtype="Int64")
        ranked_df["PreviousSignalType"] = pd.Series(dtype=str)
        ranked_df["LifecycleAction"] = pd.Series(dtype=str)
        return ranked_df

    ranked_df = ranked_df.reset_index(drop=True).copy()
    ranked_df.insert(0, "Rank", range(1, len(ranked_df) + 1))
    ranked_df["PreviousRank"] = pd.Series([pd.NA] * len(ranked_df), dtype="Int64")
    ranked_df["RankChange"] = pd.Series([pd.NA] * len(ranked_df), dtype="Int64")
    ranked_df["PreviousSignalType"] = pd.Series([pd.NA] * len(ranked_df), dtype="object")
    ranked_df["LifecycleAction"] = ranked_df.apply(
        lambda row: build_lifecycle_action(
            row["SignalType"],
            pd.NA,
            0.0 if row["BreakoutAge"] == "0d" else np.nan,
            float(row["DistFromBreakoutPct"]),
        ),
        axis=1,
    )
    return ranked_df


def apply_rank_history(ranked_df: pd.DataFrame, history_path: str = RANK_HISTORY_OUTPUT) -> pd.DataFrame:
    ranked_df = add_rank_columns(ranked_df)
    if ranked_df.empty or not Path(history_path).exists():
        return ranked_df

    history_df = load_rank_history_df(history_path)
    if history_df.empty or "RunTimestamp" not in history_df.columns:
        return ranked_df

    previous_timestamp = (
        history_df["RunTimestamp"]
        .dropna()
        .sort_values()
        .drop_duplicates()
    )
    if previous_timestamp.empty:
        return ranked_df

    last_run = previous_timestamp.iloc[-1]
    previous_snapshot = history_df[history_df["RunTimestamp"] == last_run].copy()
    previous_ranks = previous_snapshot.set_index("Symbol")["Rank"].to_dict()
    previous_signal_types = (
        previous_snapshot.set_index("Symbol")["SignalType"].to_dict()
        if "SignalType" in previous_snapshot.columns
        else {}
    )

    ranked_df["PreviousRank"] = ranked_df["Symbol"].map(previous_ranks).astype("Int64")
    ranked_df["RankChange"] = (ranked_df["PreviousRank"] - ranked_df["Rank"]).astype("Int64")
    ranked_df["PreviousSignalType"] = ranked_df["Symbol"].map(previous_signal_types)
    ranked_df["LifecycleAction"] = ranked_df.apply(
        lambda row: build_lifecycle_action(
            row["SignalType"],
            row["PreviousSignalType"],
            0.0 if row["BreakoutAge"] == "0d" else np.nan,
            float(row["DistFromBreakoutPct"]),
        ),
        axis=1,
    )
    return ranked_df


def append_snapshot(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        return
    snapshot_path = Path(path)
    existing_df = load_rank_history_df(path) if snapshot_path.exists() and path == RANK_HISTORY_OUTPUT else (
        pd.read_csv(snapshot_path) if snapshot_path.exists() else pd.DataFrame()
    )
    combined_df = pd.concat([existing_df, df], ignore_index=True, sort=False)
    if path == RANK_HISTORY_OUTPUT:
        combined_df = combined_df.reindex(columns=RANK_HISTORY_COLUMNS)
    combined_df.to_csv(snapshot_path, index=False)


def persist_rank_history(
    ranked_df: pd.DataFrame,
    blocked_df: pd.DataFrame,
    source_label: str,
    rank_history_path: str = RANK_HISTORY_OUTPUT,
    blocked_history_path: str = BLOCKED_HISTORY_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_timestamp = datetime.now().replace(microsecond=0).isoformat()
    snapshot_date = date.today().isoformat()

    ranked_with_history = apply_rank_history(ranked_df, history_path=rank_history_path)
    if not ranked_with_history.empty:
        ranked_snapshot = ranked_with_history.copy()
        ranked_snapshot.insert(0, "SnapshotDate", snapshot_date)
        ranked_snapshot.insert(1, "RunTimestamp", run_timestamp)
        ranked_snapshot.insert(2, "Source", source_label)
        append_snapshot(ranked_snapshot, rank_history_path)

    if not blocked_df.empty:
        blocked_snapshot = blocked_df.copy()
        blocked_snapshot.insert(0, "SnapshotDate", snapshot_date)
        blocked_snapshot.insert(1, "RunTimestamp", run_timestamp)
        blocked_snapshot.insert(2, "Source", source_label)
        append_snapshot(blocked_snapshot, blocked_history_path)

    return ranked_with_history, blocked_df


def build_diagnostics_frame(
    close: pd.DataFrame,
    breakout_age: pd.DataFrame,
    extension_from_ma20: pd.DataFrame,
    extension_from_breakout: pd.DataFrame,
    runup_10d_pct: pd.DataFrame,
    avg_dollar_vol_20d: pd.DataFrame,
) -> pd.DataFrame:
    return pd.concat(
        {
            symbol: pd.DataFrame(
                {
                    "breakout_age": breakout_age[symbol],
                    "extension_from_ma20": extension_from_ma20[symbol],
                    "extension_from_breakout": extension_from_breakout[symbol],
                    "runup_10d_pct": runup_10d_pct[symbol],
                    "avg_dollar_vol_20d": avg_dollar_vol_20d[symbol],
                }
            )
            for symbol in close.columns
        },
        axis=1,
    )


def run_scanner(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro_snapshot = get_macro_regime_snapshot()
    benchmark_snapshot = get_benchmark_change_snapshot()
    leveraged_flags = build_metadata_flags(symbols)

    close_dict: dict[str, pd.Series] = {}
    high_dict: dict[str, pd.Series] = {}
    low_dict: dict[str, pd.Series] = {}
    vol_dict: dict[str, pd.Series] = {}
    fetch_blocked_rows: list[dict[str, str]] = []

    for symbol in symbols:
        df = download_symbol(symbol)
        if df is None:
            fetch_blocked_rows.append({"Symbol": symbol, "blocked_reason": "fetch_failed_or_insufficient_history"})
            continue

        close_dict[symbol] = df["Close"].rename(symbol)
        high_dict[symbol] = df["High"].rename(symbol)
        low_dict[symbol] = df["Low"].rename(symbol)
        vol_dict[symbol] = df["Volume"].rename(symbol)

    if not close_dict:
        return pd.DataFrame(), pd.DataFrame(fetch_blocked_rows)

    close = pd.DataFrame(close_dict).sort_index().ffill()
    high = pd.DataFrame(high_dict).reindex(close.index).ffill()
    low = pd.DataFrame(low_dict).reindex(close.index).ffill()
    vol = pd.DataFrame(vol_dict).reindex(close.index).ffill()

    close = close.loc[:, ~close.columns.duplicated()]
    high = high.loc[:, ~high.columns.duplicated()]
    low = low.loc[:, ~low.columns.duplicated()]
    vol = vol.loc[:, ~vol.columns.duplicated()]

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()

    trend = (close > ma50) & (ma50 > ma150)
    hh50 = close.rolling(50).max()
    prior_hh50 = hh50.shift(1)
    breakout = high >= prior_hh50
    breakout_age = bars_since_true(breakout)

    close_to_breakout = close >= prior_hh50 * (1 - SETUP_PROXIMITY)
    pre_breakout = close < prior_hh50
    fresh_breakout = breakout_age <= MAX_BREAKOUT_AGE

    extension_from_ma20 = close / ma20 - 1
    extension_from_breakout = close / prior_hh50 - 1
    stock_change_20d_pct = close.pct_change(20) * 100
    runup_10d_pct = close.pct_change(10) * 100
    avg_dollar_vol_20d = (close * vol).rolling(20).mean()
    rs_vs_spy_20d = stock_change_20d_pct - benchmark_snapshot["SPY_20d_pct"]
    rs_vs_qqq_20d = stock_change_20d_pct - benchmark_snapshot["QQQ_20d_pct"]

    non_leveraged = pd.DataFrame(
        {col: not bool(leveraged_flags.get(col, False)) for col in close.columns},
        index=close.index,
    )
    price_ok = close >= MIN_PRICE
    liquidity_ok = avg_dollar_vol_20d >= MIN_AVG_DOLLAR_VOL
    runup_ok = runup_10d_pct <= MAX_10D_RUNUP_PCT
    not_extended = extension_from_ma20 <= (MAX_ABOVE_MA20_PCT / 100)
    pullback_zone = extension_from_ma20 >= -0.02
    controlled_breakout = extension_from_breakout <= MAX_BREAKOUT_EXTENSION
    rs_ok = (rs_vs_spy_20d >= MIN_RS_VS_SPY_20D) & (rs_vs_qqq_20d >= MIN_RS_VS_QQQ_20D)
    guardrails = non_leveraged & price_ok & liquidity_ok & runup_ok & rs_ok

    setup_candidate = trend & close_to_breakout & pre_breakout & not_extended & pullback_zone & guardrails
    breakout_candidate = trend & fresh_breakout & controlled_breakout & not_extended & pullback_zone & guardrails
    candidate = setup_candidate | breakout_candidate

    distance_to_breakout = ((prior_hh50 - close) / prior_hh50).clip(lower=0)
    setup_readiness = 1 - (distance_to_breakout / SETUP_PROXIMITY).clip(lower=0, upper=1)

    prev_close = close.shift(1)
    tr = pd.DataFrame(
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
    atr20 = tr.rolling(20).mean()
    atr200 = tr.rolling(200).mean()
    tightness = (1 - atr20 / atr200).clip(0, 1)

    trend_strength = ((ma50 / ma150) - 1).clip(lower=0, upper=0.20) / 0.20
    ma_stretch_score = 1 - (extension_from_ma20.abs() / (MAX_ABOVE_MA20_PCT / 100)).clip(lower=0, upper=1)
    breakout_extension_score = 1 - (
        extension_from_breakout.clip(lower=0) / MAX_BREAKOUT_EXTENSION
    ).clip(lower=0, upper=1)
    freshness_score = 1 - (breakout_age / MAX_BREAKOUT_AGE).clip(lower=0, upper=1)
    freshness_score = freshness_score.fillna(0)
    signal_type_bonus = setup_candidate.astype(float) * 1.0 + breakout_candidate.astype(float) * 0.7

    score = (
        0.28 * setup_readiness
        + 0.20 * tightness
        + 0.18 * trend_strength
        + 0.16 * ma_stretch_score
        + 0.12 * breakout_extension_score
        + 0.06 * freshness_score
        + 0.06 * signal_type_bonus
    )
    score = score * float(macro_snapshot["scanner_multiplier"])

    last_date = close.index[-1]
    today_score = score.iloc[-1].where(candidate.iloc[-1]).dropna().sort_values(ascending=False)
    diagnostics = build_diagnostics_frame(
        close,
        breakout_age,
        extension_from_ma20,
        extension_from_breakout,
        runup_10d_pct,
        avg_dollar_vol_20d,
    )
    ranked_df = format_ranked_candidates(
        today_score,
        diagnostics,
        setup_candidate,
        macro_snapshot,
        {
            "rs_vs_spy_20d": rs_vs_spy_20d.loc[last_date].to_dict(),
            "rs_vs_qqq_20d": rs_vs_qqq_20d.loc[last_date].to_dict(),
        },
    )

    blocked_rows = list(fetch_blocked_rows)
    for symbol in close.columns:
        if symbol in today_score.index:
            continue

        ordered_reasons = [
            ("leveraged_or_inverse", not bool(non_leveraged.loc[last_date, symbol])),
            ("price_below_min", not bool(price_ok.loc[last_date, symbol])),
            ("dollar_volume_below_min", not bool(liquidity_ok.loc[last_date, symbol])),
            ("runup_10d_too_hot", not bool(runup_ok.loc[last_date, symbol])),
            ("rs_vs_spy_20d_too_weak", not bool(rs_vs_spy_20d.loc[last_date, symbol] >= MIN_RS_VS_SPY_20D)),
            ("rs_vs_qqq_20d_too_weak", not bool(rs_vs_qqq_20d.loc[last_date, symbol] >= MIN_RS_VS_QQQ_20D)),
            ("not_in_trend", not bool(trend.loc[last_date, symbol])),
            ("too_extended_above_ma20", not bool(not_extended.loc[last_date, symbol])),
            ("too_far_below_ma20", not bool(pullback_zone.loc[last_date, symbol])),
            (
                "breakout_too_extended",
                bool(fresh_breakout.loc[last_date, symbol]) and not bool(controlled_breakout.loc[last_date, symbol]),
            ),
            (
                "not_near_breakout_or_fresh_breakout",
                not bool(close_to_breakout.loc[last_date, symbol]) and not bool(fresh_breakout.loc[last_date, symbol]),
            ),
        ]
        blocked_rows.append({"Symbol": symbol, "blocked_reason": last_true_reason(ordered_reasons)})

    blocked_df = pd.DataFrame(blocked_rows).sort_values("Symbol").reset_index(drop=True)
    return ranked_df, blocked_df


def run_scanner_from_csv(csv_path: str, blocked_output: str = BLOCKED_OUTPUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = load_symbols(csv_path)
    ranked_df, blocked_df = run_scanner(symbols)
    ranked_df, blocked_df = persist_rank_history(ranked_df, blocked_df, source_label=Path(csv_path).name)
    blocked_df.to_csv(blocked_output, index=False)
    return ranked_df, blocked_df


def main() -> None:
    csv_path = sys.argv[1]
    ranked_df, blocked_df = run_scanner_from_csv(csv_path)

    print("\nDate:", date.today())
    print("Ranked candidates:", len(ranked_df))
    print(f"Blocked symbols saved to {BLOCKED_OUTPUT}")
    print("")

    if ranked_df.empty:
        return

    for i, row in enumerate(ranked_df.head(20).itertuples(index=False), 1):
        print(
            f"{i:2d}. {row.Symbol:6s}  score:{row.Score:.3f}  type:{row.SignalType:8s}"
            f"  breakout_age:{row.BreakoutAge:>5s}  dist_from_MA20:{row.DistFromMA20Pct:6.2f}%"
            f"  dist_from_breakout:{row.DistFromBreakoutPct:6.2f}%  runup_10d:{row.Runup10dPct:6.2f}%"
            f"  avg_dollar_vol_20d:{row.AvgDollarVol20dM:7.1f}M"
        )


if __name__ == "__main__":
    main()
