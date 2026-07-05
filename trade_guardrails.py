from __future__ import annotations

from pathlib import Path
import sys

VENV_SITE_PACKAGES = Path(__file__).resolve().parent / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import pandas as pd
import yfinance as yf


MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOL = 20_000_000
MAX_10D_RUNUP_PCT = 15.0
MAX_ABOVE_MA20_PCT = 8.0
LEVERAGED_HINTS = (
    "ultrapro",
    "ultrashort",
    "3x",
    "2x",
    "leveraged",
    "inverse",
    "daily bull",
    "daily bear",
)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def build_metadata_flags(symbols: list[str]) -> pd.Series:
    flags: dict[str, bool] = {}

    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).info
            name = str(info.get("longName") or info.get("shortName") or "").lower()
            category = str(info.get("category") or "").lower()
            quote_type = str(info.get("quoteType") or "").lower()
            text = f"{name} {category}"
            leveraged = any(token in text for token in LEVERAGED_HINTS)
            inverse = "inverse" in text or "short" in text
            # ETFs are allowed to pass even when they are leveraged or inverse.
            # Keep the leverage guardrail only for non-ETF instruments.
            flags[symbol] = quote_type != "etf" and (leveraged or inverse)
        except Exception:
            flags[symbol] = False

    return pd.Series(flags, dtype=bool)
