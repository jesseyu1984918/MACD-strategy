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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-trade guardrails based on observed transaction mistakes.")
    parser.add_argument("symbols", nargs="+", help="Ticker symbols to evaluate")
    parser.add_argument("--side", choices=["long", "short"], default="long")
    return parser.parse_args()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def load_history(symbol: str) -> pd.DataFrame:
    hist = yf.download(symbol, period="1y", auto_adjust=False, progress=False)
    hist = flatten_columns(hist)
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    hist = hist.rename(columns=str.title)
    hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    hist["ma20"] = hist["Close"].rolling(20).mean()
    hist["ma50"] = hist["Close"].rolling(50).mean()
    hist["runup_10d_pct"] = hist["Close"].pct_change(10) * 100
    hist["avg_dollar_vol_20d"] = (hist["Close"] * hist["Volume"]).rolling(20).mean()
    return hist


def is_leveraged_or_inverse(symbol: str) -> bool:
    info = yf.Ticker(symbol).info
    name = str(info.get("longName") or info.get("shortName") or "").lower()
    category = str(info.get("category") or "").lower()
    text = f"{name} {category}"
    return any(token in text for token in LEVERAGED_HINTS)


def evaluate_symbol(symbol: str, side: str) -> dict:
    hist = load_history(symbol)
    if hist.empty:
        return {"symbol": symbol, "status": "NO_DATA", "reasons": ["No market history returned"]}

    last = hist.iloc[-1]
    reasons = []

    if last["Close"] < 5:
        reasons.append(f"Price under $5 ({last['Close']:.2f})")
    if last["avg_dollar_vol_20d"] < 20_000_000:
        reasons.append(f"20-day average dollar volume under $20M ({last['avg_dollar_vol_20d'] / 1_000_000:.1f}M)")
    if pd.notna(last["runup_10d_pct"]) and last["runup_10d_pct"] > 15:
        reasons.append(f"10-day run-up too hot ({last['runup_10d_pct']:.1f}%)")
    if pd.notna(last["ma20"]) and (last["Close"] / last["ma20"] - 1) * 100 > 8:
        reasons.append(f"More than 8% above MA20 ({(last['Close'] / last['ma20'] - 1) * 100:.1f}%)")
    if is_leveraged_or_inverse(symbol):
        reasons.append("Leveraged or inverse product")

    if side == "long":
        if pd.notna(last["ma50"]) and last["Close"] < last["ma50"]:
            reasons.append("Long setup is below the 50-day average")
    else:
        if pd.notna(last["ma20"]) and pd.notna(last["ma50"]) and last["Close"] > last["ma20"] and last["Close"] > last["ma50"]:
            reasons.append("Short setup is still above both MA20 and MA50")

    status = "BLOCK" if reasons else "PASS"
    return {
        "symbol": symbol,
        "status": status,
        "close": round(float(last["Close"]), 2),
        "ma20_gap_pct": round((last["Close"] / last["ma20"] - 1) * 100, 2) if pd.notna(last["ma20"]) else np.nan,
        "runup_10d_pct": round(float(last["runup_10d_pct"]), 2) if pd.notna(last["runup_10d_pct"]) else np.nan,
        "avg_dollar_vol_20d_m": round(float(last["avg_dollar_vol_20d"]) / 1_000_000, 2) if pd.notna(last["avg_dollar_vol_20d"]) else np.nan,
        "reasons": reasons,
    }


def main() -> None:
    args = parse_args()
    results = [evaluate_symbol(symbol.upper(), args.side) for symbol in args.symbols]

    for result in results:
        print(f"{result['symbol']}: {result['status']}")
        if result["status"] == "NO_DATA":
            print("  No data available")
            continue

        print(
            f"  Close={result['close']:.2f} MA20Gap={result['ma20_gap_pct']:.2f}% "
            f"Runup10d={result['runup_10d_pct']:.2f}% AvgDollarVol20d={result['avg_dollar_vol_20d_m']:.2f}M"
        )
        if result["reasons"]:
            for reason in result["reasons"]:
                print(f"  - {reason}")


if __name__ == "__main__":
    main()
