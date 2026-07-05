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
SETUP_MAX_ABOVE_MA20_PCT = 5.0
SETUP_MAX_ABOVE_MA50_PCT = 6.0
BREAKOUT_MAX_ABOVE_MA20_PCT = 6.0
BREAKOUT_MAX_ABOVE_MA50_PCT = 6.0
MOMENTUM_MAX_ABOVE_MA50_PCT = 6.0
LEADER_MAX_ABOVE_MA50_PCT = 6.0
MOMENTUM_MAX_ABOVE_MA20_PCT = 7.0
MOMENTUM_MAX_10D_RUNUP_PCT = 10.0
MOMENTUM_MIN_RS_VS_SPY_20D = 10.0
MOMENTUM_MIN_RS_VS_QQQ_20D = 5.0
MOMENTUM_MAX_BREAKOUT_EXTENSION = 0.003
LEADER_MAX_ABOVE_MA20_PCT = 7.0
LEADER_MAX_10D_RUNUP_PCT = 20.0
LEADER_MIN_RS_VS_SPY_20D = 10.0
LEADER_MIN_RS_VS_QQQ_20D = 5.0
LEADER_MAX_BREAKOUT_AGE = 20
LEADER_MAX_BELOW_BREAKOUT_PCT = 4.0
RECLAIM_MAX_ABOVE_MA20_PCT = 6.0
RECLAIM_MAX_10D_RUNUP_PCT = 12.0
RECLAIM_MIN_RS_VS_SPY_20D = 0.0
RECLAIM_MIN_RS_VS_QQQ_20D = -1.0
TREND_MODES = ("ma50", "ma20", "hybrid")
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


def soft_clip_ratio(series: pd.DataFrame | pd.Series, low: float, high: float) -> pd.DataFrame | pd.Series:
    return ((series - low) / (high - low)).clip(lower=0, upper=1)


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


def download_symbol(symbol: str, lookback: str = LOOKBACK) -> pd.DataFrame | None:
    try:
        df = yf.download(symbol, period=lookback, interval="1d", auto_adjust=False, progress=False)
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


def build_trend_mask(
    close: pd.DataFrame,
    ma20: pd.DataFrame,
    ma50: pd.DataFrame,
    ma150: pd.DataFrame,
    trend_mode: str = "ma50",
) -> pd.DataFrame:
    if trend_mode == "ma20":
        return (close > ma20) & (ma20 > ma50)
    if trend_mode == "hybrid":
        return (ma50 > ma150) & (close > ma20) & (ma20.diff() > 0)
    return (close > ma50) & (ma50 > ma150)


def build_ma50_extension_mask(
    extension_from_ma50: pd.DataFrame,
    trend_mode: str,
    max_extension_pct: float,
) -> pd.DataFrame:
    if trend_mode == "ma20":
        return pd.DataFrame(True, index=extension_from_ma50.index, columns=extension_from_ma50.columns)
    return extension_from_ma50 <= (max_extension_pct / 100)


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
    if signal_type == "RECLAIM":
        return "Starter only while it holds above the 20-day and keeps reclaiming trend"
    if signal_type == "MOMENTUM":
        return "Momentum entry only on controlled strength; avoid adding if it re-extends"
    if signal_type == "LEADER":
        return "Leader continuation entry only while relative strength stays dominant and extension stays controlled"

    if pd.notna(breakout_age) and breakout_age <= 0:
        return "Actionable now: starter only if the breakout is holding"

    if dist_from_breakout_pct <= 1.5:
        return "Wait for a tight hold above the breakout, then enter"

    return "Avoid chasing here; only act on a controlled pullback"


def build_trade_action(signal_type: str, breakout_age: float, dist_from_breakout_pct: float) -> str:
    if signal_type == "SETUP":
        return "WATCH"
    if signal_type == "RECLAIM":
        return "BUY"
    if signal_type == "MOMENTUM":
        return "BUY"
    if signal_type == "LEADER":
        return "BUY"

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
    if signal_type == "RECLAIM":
        if previous_signal == "RECLAIM":
            return "Reclaim still intact: only keep if price stays above the 20-day"
        return "New reclaim: review for early entry before full trend confirmation"
    if signal_type == "MOMENTUM":
        if previous_signal == "MOMENTUM":
            return "Momentum still active: only add on controlled follow-through"
        return "New momentum burst: review quickly before it gets too extended"
    if signal_type == "LEADER":
        if previous_signal == "LEADER":
            return "Leader continuation still intact: only hold or add while it stays tight"
        return "Leader continuation: review for add-on entry while strength remains orderly"

    if previous_signal == "SETUP":
        return "Triggered from prior setup: review now for entry"

    if pd.notna(breakout_age) and breakout_age <= 0:
        return "New breakout today: review now for entry"

    if previous_signal == "BREAKOUT":
        if dist_from_breakout_pct <= 1.5:
            return "Still actionable: watch for tight hold above breakout"
        return "Missed initial breakout: wait for pullback, do not chase"

    return "Fresh breakout: only act if it stays tight above the pivot"


RANKED_CANDIDATE_COLUMNS = [
    "Symbol",
    "Score",
    "SignalType",
    "TradeAction",
    "NextAction",
    "BreakoutAge",
    "DistFromMA20Pct",
    "DistFromMA50Pct",
    "DistFromBreakoutPct",
    "Runup10dPct",
    "AvgDollarVol20dM",
    "RSvsSPY20d",
    "RSvsQQQ20d",
    "MacroRegime",
    "MacroScore",
    "MacroTrend",
    "MacroRisk",
    "MacroRankAdjustment",
]


def empty_ranked_candidates_df() -> pd.DataFrame:
    return pd.DataFrame(columns=RANKED_CANDIDATE_COLUMNS)


def get_macro_signal_adjustment(signal_type: str, macro_regime: str, macro_trend: str, macro_risk: str) -> float:
    signal = str(signal_type).upper()
    regime = str(macro_regime)
    trend = str(macro_trend).lower()
    risk = str(macro_risk).lower()

    if regime == "Risk-on":
        return {
            "SETUP": 1.01,
            "BREAKOUT": 1.02,
            "RECLAIM": 1.00,
            "MOMENTUM": 1.03,
            "LEADER": 1.025,
        }.get(signal, 1.0)

    if regime == "Risk-off":
        return {
            "SETUP": 0.95,
            "BREAKOUT": 0.96,
            "RECLAIM": 0.97,
            "MOMENTUM": 0.94,
            "LEADER": 0.98,
        }.get(signal, 1.0)

    if "constructive" in risk and "supportive" in trend:
        return {
            "SETUP": 1.005,
            "BREAKOUT": 1.01,
            "RECLAIM": 1.00,
            "MOMENTUM": 1.015,
            "LEADER": 1.015,
        }.get(signal, 1.0)

    if "fragile" in trend or "defensive" in risk or "risk-off" in risk:
        return {
            "SETUP": 0.98,
            "BREAKOUT": 0.985,
            "RECLAIM": 0.99,
            "MOMENTUM": 0.98,
            "LEADER": 0.995,
        }.get(signal, 1.0)

    return 1.0


def build_macro_signal_adjustment(
    signal_type: pd.DataFrame,
    macro_regime: str,
    macro_trend: str,
    macro_risk: str,
) -> pd.DataFrame:
    adjustment = pd.DataFrame(1.0, index=signal_type.index, columns=signal_type.columns)
    for signal_name in ["SETUP", "BREAKOUT", "RECLAIM", "MOMENTUM", "LEADER"]:
        adjustment = adjustment.mask(
            signal_type.eq(signal_name),
            get_macro_signal_adjustment(signal_name, macro_regime, macro_trend, macro_risk),
        )
    return adjustment


def format_ranked_candidates(
    today_score: pd.Series,
    diagnostics: pd.DataFrame,
    signal_type: pd.DataFrame,
    macro_snapshot: dict[str, object],
    rs_snapshot: dict[str, object],
) -> pd.DataFrame:
    if today_score.empty:
        return empty_ranked_candidates_df()

    last_date = diagnostics.index[-1]
    rows = []

    for symbol, val in today_score.items():
        age = diagnostics.at[last_date, (symbol, "breakout_age")]
        signal_name = str(signal_type.at[last_date, symbol])
        dist_from_breakout_pct = round(float(diagnostics.at[last_date, (symbol, "extension_from_breakout")]) * 100, 2)
        rows.append(
            {
                "Symbol": symbol,
                "Score": round(float(val), 3),
                "SignalType": signal_name,
                "TradeAction": build_trade_action(signal_name, age, dist_from_breakout_pct),
                "NextAction": build_next_action(signal_name, age, dist_from_breakout_pct),
                "BreakoutAge": "setup" if pd.isna(age) else f"{int(age)}d",
                "DistFromMA20Pct": round(float(diagnostics.at[last_date, (symbol, "extension_from_ma20")]) * 100, 2),
                "DistFromMA50Pct": round(float(diagnostics.at[last_date, (symbol, "extension_from_ma50")]) * 100, 2),
                "DistFromBreakoutPct": dist_from_breakout_pct,
                "Runup10dPct": round(float(diagnostics.at[last_date, (symbol, "runup_10d_pct")]), 2),
                "AvgDollarVol20dM": round(float(diagnostics.at[last_date, (symbol, "avg_dollar_vol_20d")]) / 1_000_000, 2),
                "RSvsSPY20d": round(float(rs_snapshot["rs_vs_spy_20d"].get(symbol, np.nan)), 2),
                "RSvsQQQ20d": round(float(rs_snapshot["rs_vs_qqq_20d"].get(symbol, np.nan)), 2),
                "MacroRegime": str(macro_snapshot["regime"]),
                "MacroScore": int(macro_snapshot["score"]),
                "MacroTrend": str(macro_snapshot["trend"]),
                "MacroRisk": str(macro_snapshot["risk"]),
                "MacroRankAdjustment": round(
                    get_macro_signal_adjustment(
                        signal_name,
                        str(macro_snapshot["regime"]),
                        str(macro_snapshot["trend"]),
                        str(macro_snapshot["risk"]),
                    ),
                    3,
                ),
            }
        )

    return pd.DataFrame(rows, columns=RANKED_CANDIDATE_COLUMNS)


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
    extension_from_ma50: pd.DataFrame,
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
                    "extension_from_ma50": extension_from_ma50[symbol],
                    "extension_from_breakout": extension_from_breakout[symbol],
                    "runup_10d_pct": runup_10d_pct[symbol],
                    "avg_dollar_vol_20d": avg_dollar_vol_20d[symbol],
                }
            )
            for symbol in close.columns
        },
        axis=1,
    )


def run_scanner(symbols: list[str], trend_mode: str = "ma50") -> tuple[pd.DataFrame, pd.DataFrame]:
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
        return empty_ranked_candidates_df(), pd.DataFrame(fetch_blocked_rows)

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

    trend = build_trend_mask(close, ma20, ma50, ma150, trend_mode=trend_mode)
    hh50 = close.rolling(50).max()
    prior_hh50 = hh50.shift(1)
    breakout = high >= prior_hh50
    breakout_age = bars_since_true(breakout)

    close_to_breakout = close >= prior_hh50 * (1 - SETUP_PROXIMITY)
    pre_breakout = close < prior_hh50
    fresh_breakout = breakout_age <= MAX_BREAKOUT_AGE

    extension_from_ma20 = close / ma20 - 1
    extension_from_ma50 = close / ma50 - 1
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
    setup_ma50_not_extended = build_ma50_extension_mask(extension_from_ma50, trend_mode, SETUP_MAX_ABOVE_MA50_PCT)
    breakout_ma50_not_extended = build_ma50_extension_mask(extension_from_ma50, trend_mode, BREAKOUT_MAX_ABOVE_MA50_PCT)
    momentum_ma50_not_extended = build_ma50_extension_mask(extension_from_ma50, trend_mode, MOMENTUM_MAX_ABOVE_MA50_PCT)
    leader_ma50_not_extended = build_ma50_extension_mask(extension_from_ma50, trend_mode, LEADER_MAX_ABOVE_MA50_PCT)
    pullback_zone = extension_from_ma20 >= -0.02
    controlled_breakout = extension_from_breakout <= MAX_BREAKOUT_EXTENSION
    rs_ok = (rs_vs_spy_20d >= MIN_RS_VS_SPY_20D) & (rs_vs_qqq_20d >= MIN_RS_VS_QQQ_20D)
    guardrails = non_leveraged & price_ok & liquidity_ok & runup_ok & rs_ok

    setup_candidate = (
        trend
        & close_to_breakout
        & pre_breakout
        & (extension_from_ma20 <= (SETUP_MAX_ABOVE_MA20_PCT / 100))
        & setup_ma50_not_extended
        & pullback_zone
        & guardrails
    )
    breakout_candidate = (
        trend
        & fresh_breakout
        & (close >= prior_hh50)
        & controlled_breakout
        & (extension_from_ma20 <= (BREAKOUT_MAX_ABOVE_MA20_PCT / 100))
        & breakout_ma50_not_extended
        & pullback_zone
        & guardrails
    )
    momentum_candidate = (
        trend
        & fresh_breakout
        & (close >= prior_hh50)
        & (extension_from_breakout <= MOMENTUM_MAX_BREAKOUT_EXTENSION)
        & (extension_from_ma20 <= (MOMENTUM_MAX_ABOVE_MA20_PCT / 100))
        & momentum_ma50_not_extended
        & (runup_10d_pct <= MOMENTUM_MAX_10D_RUNUP_PCT)
        & (rs_vs_spy_20d >= MOMENTUM_MIN_RS_VS_SPY_20D)
        & (rs_vs_qqq_20d >= MOMENTUM_MIN_RS_VS_QQQ_20D)
        & non_leveraged
        & price_ok
        & liquidity_ok
    )
    leader_candidate = (
        trend
        & (breakout_age <= LEADER_MAX_BREAKOUT_AGE)
        & (close >= prior_hh50)
        & (extension_from_breakout >= -(LEADER_MAX_BELOW_BREAKOUT_PCT / 100))
        & (close > ma20)
        & (extension_from_ma20 >= 0)
        & (extension_from_ma20 <= (LEADER_MAX_ABOVE_MA20_PCT / 100))
        & leader_ma50_not_extended
        & (runup_10d_pct <= LEADER_MAX_10D_RUNUP_PCT)
        & (rs_vs_spy_20d >= LEADER_MIN_RS_VS_SPY_20D)
        & (rs_vs_qqq_20d >= LEADER_MIN_RS_VS_QQQ_20D)
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
        & (extension_from_ma20 <= (RECLAIM_MAX_ABOVE_MA20_PCT / 100))
        & setup_ma50_not_extended
        & (runup_10d_pct <= RECLAIM_MAX_10D_RUNUP_PCT)
        & (rs_vs_spy_20d >= RECLAIM_MIN_RS_VS_SPY_20D)
        & (rs_vs_qqq_20d >= RECLAIM_MIN_RS_VS_QQQ_20D)
        & non_leveraged
        & price_ok
        & liquidity_ok
    )
    momentum_candidate = momentum_candidate & ~(setup_candidate | breakout_candidate)
    leader_candidate = leader_candidate & ~(setup_candidate | breakout_candidate | momentum_candidate)
    reclaim_candidate = reclaim_candidate & ~(setup_candidate | breakout_candidate | momentum_candidate | leader_candidate)
    candidate = setup_candidate | breakout_candidate | momentum_candidate | leader_candidate | reclaim_candidate

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
    atr100 = tr.rolling(100).mean()
    atr60 = tr.rolling(60).mean()
    atr_baseline = atr200.combine_first(atr100).combine_first(atr60)
    tightness = (1 - atr20 / atr_baseline).clip(0, 1)

    trend_strength = ((ma50 / ma150) - 1).clip(lower=0, upper=0.20) / 0.20
    ma_stretch_score = 1 - (extension_from_ma20.abs() / (MAX_ABOVE_MA20_PCT / 100)).clip(lower=0, upper=1)
    breakout_extension_score = 1 - (
        extension_from_breakout.clip(lower=0) / MAX_BREAKOUT_EXTENSION
    ).clip(lower=0, upper=1)
    freshness_score = 1 - (breakout_age / MAX_BREAKOUT_AGE).clip(lower=0, upper=1)
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
        ((rs_vs_spy_20d - RECLAIM_MIN_RS_VS_SPY_20D).clip(lower=0, upper=10) / 10)
        + ((rs_vs_qqq_20d - RECLAIM_MIN_RS_VS_QQQ_20D).clip(lower=0, upper=10) / 10)
    ) / 2
    reclaim_extension_score = 1 - (
        (extension_from_ma20 - 0.02).abs() / (RECLAIM_MAX_ABOVE_MA20_PCT / 100)
    ).clip(lower=0, upper=1)
    reclaim_trigger_score = ((close / close.rolling(10).max().shift(1)) - 1).clip(lower=0, upper=0.03) / 0.03
    reclaim_score = (
        0.30 * reclaim_trend_strength
        + 0.30 * reclaim_rs_strength
        + 0.25 * reclaim_extension_score
        + 0.15 * reclaim_trigger_score
    )
    reclaim_readiness = (
        0.25 * soft_clip_ratio(ma50 / ma150, 0.95, 1.01)
        + 0.20 * soft_clip_ratio(close / ma20, 0.99, 1.02)
        + 0.15 * soft_clip_ratio(ma20 / ma20.shift(3), 0.995, 1.01)
        + 0.20 * soft_clip_ratio(rs_vs_spy_20d, -2.0, 5.0)
        + 0.20 * soft_clip_ratio(rs_vs_qqq_20d, -3.0, 3.0)
    )

    momentum_rs_strength = (
        ((rs_vs_spy_20d - MOMENTUM_MIN_RS_VS_SPY_20D).clip(lower=0, upper=30) / 30)
        + ((rs_vs_qqq_20d - MOMENTUM_MIN_RS_VS_QQQ_20D).clip(lower=0, upper=25) / 25)
    ) / 2
    momentum_extension_score = 1 - (
        (extension_from_ma20 - 0.10).abs() / (MOMENTUM_MAX_ABOVE_MA20_PCT / 100)
    ).clip(lower=0, upper=1)
    momentum_score = (
        0.28 * trend_strength
        + 0.32 * momentum_rs_strength
        + 0.20 * momentum_extension_score
        + 0.10 * breakout_extension_score
        + 0.10 * freshness_score
    )
    momentum_readiness = (
        0.25 * soft_clip_ratio(rs_vs_spy_20d, 5.0, 20.0)
        + 0.20 * soft_clip_ratio(rs_vs_qqq_20d, 0.0, 15.0)
        + 0.20 * soft_clip_ratio(MAX_BREAKOUT_AGE - breakout_age, -2.0, 2.0)
        + 0.15 * soft_clip_ratio(MOMENTUM_MAX_10D_RUNUP_PCT - runup_10d_pct, -10.0, 8.0)
        + 0.20 * (1 - ((extension_from_ma20 - 0.10).abs() / 0.15).clip(lower=0, upper=1))
    )

    leader_rs_strength = (
        ((rs_vs_spy_20d - LEADER_MIN_RS_VS_SPY_20D).clip(lower=0, upper=20) / 20)
        + ((rs_vs_qqq_20d - LEADER_MIN_RS_VS_QQQ_20D).clip(lower=0, upper=15) / 15)
    ) / 2
    leader_extension_score = 1 - (
        (extension_from_ma20 - 0.09).abs() / 0.10
    ).clip(lower=0, upper=1)
    leader_proximity_score = 1 - (((-extension_from_breakout).clip(lower=0)) / 0.30).clip(lower=0, upper=1)
    leader_persistence_score = 1 - (breakout_age / LEADER_MAX_BREAKOUT_AGE).clip(lower=0, upper=1)
    leader_score = (
        0.18 * trend_strength
        + 0.40 * leader_rs_strength
        + 0.18 * leader_extension_score
        + 0.10 * leader_proximity_score
        + 0.08 * leader_persistence_score
        + 0.12
    )
    leader_readiness = (
        0.18 * soft_clip_ratio(ma50 / ma150, 0.98, 1.06)
        + 0.30 * soft_clip_ratio(rs_vs_spy_20d, 2.0, 18.0)
        + 0.20 * soft_clip_ratio(rs_vs_qqq_20d, 0.0, 12.0)
        + 0.17 * (1 - ((extension_from_ma20 - 0.09).abs() / 0.14).clip(lower=0, upper=1))
        + 0.08 * (1 - (((-extension_from_breakout).clip(lower=0)) / 0.35).clip(lower=0, upper=1))
        + 0.07 * soft_clip_ratio(LEADER_MAX_BREAKOUT_AGE - breakout_age, 0.0, 25.0)
    )

    score = pd.concat(
        [
            base_score.stack(dropna=False).rename("base"),
            (reclaim_score * reclaim_readiness).stack(dropna=False).rename("reclaim"),
            (momentum_score * momentum_readiness).stack(dropna=False).rename("momentum"),
            (leader_score * leader_readiness).stack(dropna=False).rename("leader"),
        ],
        axis=1,
    ).max(axis=1).unstack()
    score = score * float(macro_snapshot["scanner_multiplier"])
    score = score * build_macro_signal_adjustment(
        signal_type,
        str(macro_snapshot["regime"]),
        str(macro_snapshot["trend"]),
        str(macro_snapshot["risk"]),
    )

    last_date = close.index[-1]
    today_score = score.iloc[-1].where(candidate.iloc[-1]).dropna().sort_values(ascending=False)
    diagnostics = build_diagnostics_frame(
        close,
        breakout_age,
        extension_from_ma20,
        extension_from_ma50,
        extension_from_breakout,
        runup_10d_pct,
        avg_dollar_vol_20d,
    )
    ranked_df = format_ranked_candidates(
        today_score,
        diagnostics,
        signal_type,
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
            (
                "too_extended_above_ma50",
                not bool(setup_ma50_not_extended.loc[last_date, symbol]) and not bool(breakout_ma50_not_extended.loc[last_date, symbol]),
            ),
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
        blocked_reason = last_true_reason(ordered_reasons)
        if blocked_reason == "passed":
            # The symbol cleared the generic guardrails but still failed every
            # signal template, so "passed" is misleading in blocked output.
            blocked_reason = "no_signal_state_matched"
        blocked_rows.append({"Symbol": symbol, "blocked_reason": blocked_reason})

    blocked_df = pd.DataFrame(blocked_rows, columns=["Symbol", "blocked_reason"])
    if not blocked_df.empty:
        blocked_df = blocked_df.sort_values("Symbol").reset_index(drop=True)
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
            f"  dist_from_MA50:{row.DistFromMA50Pct:6.2f}%"
            f"  dist_from_breakout:{row.DistFromBreakoutPct:6.2f}%  runup_10d:{row.Runup10dPct:6.2f}%"
            f"  avg_dollar_vol_20d:{row.AvgDollarVol20dM:7.1f}M"
        )


if __name__ == "__main__":
    main()
