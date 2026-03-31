from __future__ import annotations

from pathlib import Path
import sys

VENV_SITE_PACKAGES = Path(__file__).resolve().parent / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import pandas as pd
import yfinance as yf
from yfinance import cache as yf_cache

from trade_guardrails import flatten_columns


YF_CACHE_DIR = Path(__file__).resolve().parent / ".yf_cache"
MACRO_SYMBOLS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "DIA": "Dow Jones",
    "TLT": "Long Treasuries",
    "HYG": "High Yield Credit",
    "^VIX": "VIX",
}


def download_macro_history(symbol: str) -> pd.DataFrame | None:
    try:
        YF_CACHE_DIR.mkdir(exist_ok=True)
        yf_cache.set_cache_location(str(YF_CACHE_DIR))
        hist = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
        hist = flatten_columns(hist)
        if hist is None or hist.empty:
            return None
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist = hist.rename(columns=str.title)
        hist = hist[["Close"]].dropna()
        if hist.empty:
            return None

        hist["ma20"] = hist["Close"].rolling(20).mean()
        hist["ma50"] = hist["Close"].rolling(50).mean()
        hist["ma150"] = hist["Close"].rolling(150).mean()
        hist["change_20d_pct"] = hist["Close"].pct_change(20) * 100

        ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
        ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist["macd_hist"] = macd - signal
        return hist
    except Exception:
        return None


def classify_market_row(symbol: str, row: pd.Series) -> str:
    if pd.isna(row["ma50"]) or pd.isna(row["macd_hist"]):
        return "Insufficient history"

    if symbol == "^VIX":
        if row["Close"] < row["ma50"]:
            return "Calm volatility backdrop"
        return "Elevated volatility backdrop"

    above_ma50 = row["Close"] > row["ma50"]
    above_ma150 = row["Close"] > row["ma150"] if pd.notna(row["ma150"]) else False
    macd_positive = row["macd_hist"] > 0

    if above_ma50 and above_ma150 and macd_positive:
        return "Bullish trend"
    if above_ma50 and above_ma150:
        return "Uptrend, momentum soft"
    if above_ma50 or macd_positive:
        return "Mixed"
    return "Weak"


def build_macro_status_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for symbol, label in MACRO_SYMBOLS.items():
        hist = download_macro_history(symbol)
        if hist is None or hist.empty:
            rows.append(
                {
                    "Symbol": symbol,
                    "Asset": label,
                    "Status": "No data",
                }
            )
            continue

        last = hist.iloc[-1]
        dist_from_ma20 = ((last["Close"] / last["ma20"]) - 1) * 100 if pd.notna(last["ma20"]) and last["ma20"] else None
        dist_from_ma50 = ((last["Close"] / last["ma50"]) - 1) * 100 if pd.notna(last["ma50"]) and last["ma50"] else None
        rows.append(
            {
                "Symbol": symbol,
                "Asset": label,
                "LastPrice": round(float(last["Close"]), 2),
                "Change20dPct": round(float(last["change_20d_pct"]), 2) if pd.notna(last["change_20d_pct"]) else None,
                "DistFromMA20Pct": round(float(dist_from_ma20), 2) if dist_from_ma20 is not None else None,
                "DistFromMA50Pct": round(float(dist_from_ma50), 2) if dist_from_ma50 is not None else None,
                "MACDHist": round(float(last["macd_hist"]), 4) if pd.notna(last["macd_hist"]) else None,
                "Status": classify_market_row(symbol, last),
            }
        )

    return pd.DataFrame(rows)


def build_macro_trend_df(lookback_days: int = 60) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}

    for symbol, label in MACRO_SYMBOLS.items():
        hist = download_macro_history(symbol)
        if hist is None or hist.empty:
            continue

        recent = hist["Close"].tail(lookback_days).copy()
        if recent.empty:
            continue

        series_map[label] = recent

    if not series_map:
        return pd.DataFrame()

    return pd.DataFrame(series_map)


def build_macro_summary(macro_df: pd.DataFrame) -> dict[str, str]:
    if macro_df.empty:
        return {
            "regime": "No data",
            "trend": "No data",
            "risk": "No data",
        }

    trend_symbols = ["SPY", "QQQ", "IWM", "DIA"]
    trend_df = macro_df[macro_df["Symbol"].isin(trend_symbols)].copy()
    bullish_count = int((trend_df["Status"] == "Bullish trend").sum())
    weak_count = int((trend_df["Status"] == "Weak").sum())

    if bullish_count >= 3:
        trend = "Broad equity trend is supportive"
    elif weak_count >= 2:
        trend = "Broad equity trend is fragile"
    else:
        trend = "Broad equity trend is mixed"

    risk = "Neutral"
    vix_row = macro_df[macro_df["Symbol"] == "^VIX"]
    hyg_row = macro_df[macro_df["Symbol"] == "HYG"]
    tlt_row = macro_df[macro_df["Symbol"] == "TLT"]

    if not vix_row.empty and "Elevated volatility backdrop" in vix_row["Status"].tolist():
        risk = "Risk-off pressure from volatility"
    if not hyg_row.empty and not tlt_row.empty:
        hyg_20d = hyg_row["Change20dPct"].iloc[0]
        tlt_20d = tlt_row["Change20dPct"].iloc[0]
        if pd.notna(hyg_20d) and pd.notna(tlt_20d):
            if hyg_20d > tlt_20d and risk == "Neutral":
                risk = "Risk appetite is leaning constructive"
            elif tlt_20d > hyg_20d:
                risk = "Defensive assets are leading"

    if "supportive" in trend.lower() and "constructive" in risk.lower():
        regime = "Risk-on"
    elif "fragile" in trend.lower() or "defensive" in risk.lower() or "risk-off" in risk.lower():
        regime = "Risk-off"
    else:
        regime = "Mixed"

    return {
        "regime": regime,
        "trend": trend,
        "risk": risk,
    }


def get_macro_regime_snapshot() -> dict[str, object]:
    macro_df = build_macro_status_df()
    summary = build_macro_summary(macro_df)
    regime = summary["regime"]

    if regime == "Risk-on":
        score = 1
        scanner_multiplier = 1.05
        exit_review_threshold_shift = -0.03
    elif regime == "Risk-off":
        score = -1
        scanner_multiplier = 0.88
        exit_review_threshold_shift = 0.05
    else:
        score = 0
        scanner_multiplier = 1.0
        exit_review_threshold_shift = 0.0

    return {
        "regime": regime,
        "trend": summary["trend"],
        "risk": summary["risk"],
        "score": score,
        "scanner_multiplier": scanner_multiplier,
        "exit_review_threshold_shift": exit_review_threshold_shift,
    }


def get_benchmark_change_snapshot() -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for symbol in ["SPY", "QQQ"]:
        hist = download_macro_history(symbol)
        if hist is None or hist.empty or "change_20d_pct" not in hist.columns:
            snapshot[f"{symbol}_20d_pct"] = float("nan")
            continue
        last = hist.iloc[-1]
        snapshot[f"{symbol}_20d_pct"] = float(last["change_20d_pct"]) if pd.notna(last["change_20d_pct"]) else float("nan")
    return snapshot
