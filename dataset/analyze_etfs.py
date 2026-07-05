from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


LEVERAGED = {"TQQQ", "SQQQ", "UPRO", "SOXL", "SOXS"}
CRYPTO = {"IBIT", "FBTC", "ETHA"}
COMMODITY = {"GLD", "IAU", "SLV", "DBC", "PDBC", "USO", "UNG", "GDX"}
BOND = {
    "BND", "AGG", "BSV", "SGOV", "SHY", "IEF", "TLT", "TIP", "LQD",
    "VCIT", "HYG", "JNK", "EMB", "MUB", "BNDX", "BIL",
}
CORE = {
    "SPY", "VOO", "IVV", "SPLG", "VTI", "ITOT", "SCHB", "VT", "ACWI",
    "RSP", "QQQ", "VUG", "SCHG", "IWF", "VTV", "IWD", "QUAL", "MOAT",
    "MTUM", "USMV", "IJH", "VO", "IJR", "VB", "IWM", "AVUV", "SCHA",
    "VBR", "VXUS", "IXUS", "VEA", "IEFA", "VWO", "IEMG", "EFA", "EEM",
    "SCHD", "VYM", "DGRO",
}


def category(symbol: str) -> str:
    if symbol in LEVERAGED:
        return "Leveraged/Inverse"
    if symbol in CRYPTO:
        return "Crypto"
    if symbol in COMMODITY:
        return "Commodity"
    if symbol in BOND:
        return "Bond"
    if symbol in CORE:
        return "Core/Diversified"
    return "Sector/Thematic"


def period_return(series: pd.Series, sessions: int) -> float:
    if len(series) < 2:
        return np.nan
    start = series.iloc[-min(sessions + 1, len(series))]
    return float((series.iloc[-1] / start - 1) * 100)


def calculate_metrics(symbol: str, prices: pd.Series, spy: pd.Series) -> dict:
    prices = prices.dropna()
    daily = prices.pct_change().dropna()
    aligned = pd.concat([daily, spy.pct_change()], axis=1).dropna()
    aligned.columns = ["fund", "spy"]

    annual_return = period_return(prices, 252)
    volatility = float(daily.std() * np.sqrt(252) * 100)
    running_high = prices.cummax()
    max_drawdown = float(((prices / running_high) - 1).min() * 100)
    sharpe = ((annual_return / 100) - 0.04) / (volatility / 100) if volatility else np.nan
    beta = (
        float(aligned["fund"].cov(aligned["spy"]) / aligned["spy"].var())
        if len(aligned) > 20 and aligned["spy"].var()
        else np.nan
    )

    ma50 = prices.rolling(50).mean().iloc[-1]
    ma200 = prices.rolling(200).mean().iloc[-1]
    trend = "Bullish" if prices.iloc[-1] > ma50 > ma200 else (
        "Mixed" if prices.iloc[-1] > ma200 else "Bearish"
    )
    score = (
        0.30 * period_return(prices, 63)
        + 0.25 * period_return(prices, 126)
        + 0.20 * annual_return
        + 8.0 * sharpe
        + 0.15 * max_drawdown
        - 0.08 * volatility
        + (4 if trend == "Bullish" else -4 if trend == "Bearish" else 0)
    )

    return {
        "Symbol": symbol,
        "Category": category(symbol),
        "Price": round(float(prices.iloc[-1]), 2),
        "Return1M": round(period_return(prices, 21), 2),
        "Return3M": round(period_return(prices, 63), 2),
        "Return6M": round(period_return(prices, 126), 2),
        "Return1Y": round(annual_return, 2),
        "Volatility": round(volatility, 2),
        "MaxDrawdown": round(max_drawdown, 2),
        "Sharpe": round(sharpe, 2),
        "Beta": round(beta, 2),
        "Trend": trend,
        "Score": round(score, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="ETF.csv")
    parser.add_argument("--output", default="etf_analysis.csv")
    args = parser.parse_args()

    universe = pd.read_csv(args.input)
    descriptions = dict(zip(universe["Symbol"], universe["Description"]))
    symbols = universe["Symbol"].drop_duplicates().tolist()
    download_symbols = list(dict.fromkeys(symbols + ["SPY"]))
    data = yf.download(
        download_symbols,
        period="14mo",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if not isinstance(close, pd.DataFrame):
        close = close.to_frame()

    rows = []
    errors = []
    for symbol in symbols:
        if symbol not in close or close[symbol].dropna().empty:
            errors.append(symbol)
            continue
        row = calculate_metrics(symbol, close[symbol], close["SPY"])
        row["Description"] = descriptions[symbol]
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("Score", ascending=False)
    non_leveraged = result["Category"].ne("Leveraged/Inverse")
    result["Percentile"] = np.nan
    result.loc[non_leveraged, "Percentile"] = (
        result.loc[non_leveraged, "Score"].rank(pct=True) * 100
    ).round(1)
    result["Signal"] = "HOLD"
    result.loc[non_leveraged & result["Percentile"].ge(80) & result["Trend"].eq("Bullish"), "Signal"] = "BUY"
    result.loc[non_leveraged & (result["Percentile"].le(20) | result["Trend"].eq("Bearish")), "Signal"] = "AVOID"
    result.loc[result["Category"].eq("Leveraged/Inverse"), "Signal"] = "TRADING ONLY"

    columns = [
        "Symbol", "Description", "Category", "Signal", "Score", "Percentile",
        "Price", "Return1M", "Return3M", "Return6M", "Return1Y",
        "Volatility", "MaxDrawdown", "Sharpe", "Beta", "Trend",
    ]
    result[columns].to_csv(args.output, index=False)
    result[result["Category"].eq("Core/Diversified")][columns].to_csv(
        "etf_core_rankings.csv", index=False
    )
    result[
        ~result["Category"].isin(["Core/Diversified", "Leveraged/Inverse"])
    ][columns].to_csv("etf_satellite_rankings.csv", index=False)
    print(f"Analyzed {len(result)}/{len(symbols)} ETFs; errors={','.join(errors) or 'none'}")


if __name__ == "__main__":
    main()
