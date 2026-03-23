from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

VENV_SITE_PACKAGES = Path(__file__).resolve().parent / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import numpy as np
import pandas as pd
import yfinance as yf

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
BLOCKED_OUTPUT = "scanner_blocked.csv"


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


def format_ranked_candidates(today_score: pd.Series, diagnostics: pd.DataFrame, setup_candidate: pd.DataFrame) -> pd.DataFrame:
    if today_score.empty:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Score",
                "SignalType",
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
        rows.append(
            {
                "Symbol": symbol,
                "Score": round(float(val), 3),
                "SignalType": "SETUP" if bool(setup_candidate.at[last_date, symbol]) else "BREAKOUT",
                "BreakoutAge": "setup" if pd.isna(age) else f"{int(age)}d",
                "DistFromMA20Pct": round(float(diagnostics.at[last_date, (symbol, "extension_from_ma20")]) * 100, 2),
                "DistFromBreakoutPct": round(float(diagnostics.at[last_date, (symbol, "extension_from_breakout")]) * 100, 2),
                "Runup10dPct": round(float(diagnostics.at[last_date, (symbol, "runup_10d_pct")]), 2),
                "AvgDollarVol20dM": round(float(diagnostics.at[last_date, (symbol, "avg_dollar_vol_20d")]) / 1_000_000, 2),
            }
        )

    return pd.DataFrame(rows)


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
    runup_10d_pct = close.pct_change(10) * 100
    avg_dollar_vol_20d = (close * vol).rolling(20).mean()

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
    guardrails = non_leveraged & price_ok & liquidity_ok & runup_ok

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
    ranked_df = format_ranked_candidates(today_score, diagnostics, setup_candidate)

    blocked_rows = list(fetch_blocked_rows)
    for symbol in close.columns:
        if symbol in today_score.index:
            continue

        ordered_reasons = [
            ("leveraged_or_inverse", not bool(non_leveraged.loc[last_date, symbol])),
            ("price_below_min", not bool(price_ok.loc[last_date, symbol])),
            ("dollar_volume_below_min", not bool(liquidity_ok.loc[last_date, symbol])),
            ("runup_10d_too_hot", not bool(runup_ok.loc[last_date, symbol])),
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
