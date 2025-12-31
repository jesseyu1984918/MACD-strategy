# build_dataset.py
import os
import pandas as pd
import numpy as np
import yfinance as yf
import csv
from datetime import datetime

CSV1 = "nasdaq_100_spy_500.csv"
CSV2 = "my_universe.csv"
TICKER_COL = "Symbol"

OUT_PATH = "data/weekly_ml_dataset.parquet"
os.makedirs("data", exist_ok=True)

BAD_PATH = "data/bad_symbols.csv"

def log_bad(ticker: str, reason: str):
    os.makedirs("data", exist_ok=True)
    exists = os.path.exists(BAD_PATH)
    with open(BAD_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "ticker", "reason"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), ticker, reason])


def load_all_tickers() -> list[str]:
    df1 = pd.read_csv(CSV1)
    df2 = pd.read_csv(CSV2)
    tickers = pd.concat([df1[TICKER_COL], df2[TICKER_COL]], ignore_index=True).dropna()
    tickers = tickers.astype(str).str.strip().str.upper()
    tickers = tickers[tickers != ""]
    return sorted(tickers.unique())

def udr(close: pd.Series, n: int = 10) -> float:
    if len(close) < n + 1:
        return np.nan
    last = close.iloc[-(n+1):]
    return float((last.diff() > 0).sum()) / n

def tps(hist: pd.DataFrame, n: int = 10) -> float:
    if len(hist) < 20 + n:
        return np.nan
    close = hist["Close"].astype(float)
    low = hist["Low"].astype(float)
    ma20 = close.rolling(20).mean()
    valid = pd.concat([low, ma20], axis=1).dropna()
    valid.columns = ["Low", "MA20"]
    if len(valid) < n:
        return np.nan
    recent = valid.tail(n)
    return float((recent["Low"] >= recent["MA20"]).mean())

def ma20_slope_5d(close: pd.Series) -> float:
    ma20 = close.rolling(20).mean().dropna()
    if len(ma20) < 6:
        return np.nan
    y = ma20.iloc[-5:].to_numpy()
    x = np.arange(5)
    return float(np.polyfit(x, y, 1)[0])

def vol_ratio(close: pd.Series) -> float:
    r = close.pct_change().dropna()
    if len(r) < 21:
        return np.nan
    v5 = float(r.tail(5).std())
    v20 = float(r.tail(20).std())
    return np.nan if v20 == 0 else v5 / v20

def context_label(close: pd.Series) -> int:
    # Pullback=1, Breakout=0, Other=-1 (computed at Friday)
    ma20 = close.rolling(20).mean()
    if len(close.dropna()) < 26:
        return -1
    close0, ma0 = float(close.iloc[-1]), float(ma20.iloc[-1])
    close_k, ma_k = float(close.iloc[-6]), float(ma20.iloc[-6])  # 5 trading days ago
    dist_pct = (close0 / ma0 - 1) * 100 if ma0 else np.nan

    if (close0 > ma0) and (close_k <= ma_k):
        return 0  # Breakout
    if (close0 > ma0) and (close_k > ma_k) and (dist_pct <= 2.0):
        return 1  # Pullback
    return -1

def build_rows_for_ticker(
    ticker: str,
    period: str = "5y",
    min_history_days: int = 260,
    forward_days: int = 5,
    debug: bool = False,
) -> list[dict]:
    """
    Builds weekly (Friday) rows for one ticker.
    - Week endpoint is Friday label, but uses the last actual trading day <= Friday.
    - Features computed using data up to that trading day.
    - Labels use forward trading-day indexing (t+5 trading days).
    """

    def dprint(*args):
        if debug:
            print(*args)

    # 1) Download daily OHLCV
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    ).dropna()

    # If columns are MultiIndex like ('Close','AME'), extract the ticker slice
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer slicing by ticker level (level=1)
        if ticker in df.columns.get_level_values(1):
            df = df.xs(ticker, axis=1, level=1)
        else:
            # Fallback: just keep field names (level 0)
            df.columns = df.columns.get_level_values(0)

    if df is None or df.empty:
        dprint(f"[{ticker}] no data returned")
        return []

    # Make sure index is sorted and tz-naive
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)

    # yfinance sometimes returns weird column shapes; enforce single-level columns
    # We need at least these:
    required = {"Close", "Low"}
    if not required.issubset(set(df.columns)):
        dprint(f"[{ticker}] missing columns. have={list(df.columns)}")
        return []

    close_all = df["Close"]
    low_all = df["Low"]

    # Handle case where df["Close"] is a DataFrame (multi-column); take first column
    if isinstance(close_all, pd.DataFrame):
        close_all = close_all.iloc[:, 0]
    if isinstance(low_all, pd.DataFrame):
        low_all = low_all.iloc[:, 0]

    close_all = close_all.astype(float)
    low_all = low_all.astype(float)

    dprint(f"[{ticker}] raw rows={len(df)} range={df.index.min().date()}→{df.index.max().date()}")

    # 2) Weekly Friday labels
    # Use close series just for the index reference
    week_labels = close_all.resample("W-FRI").last().dropna().index
    dprint(f"[{ticker}] weekly labels={len(week_labels)}")

    rows: list[dict] = []

    # 3) Iterate over weekly labels; map to real trading day <= label
    for t in week_labels:
        # Find last trading day <= t
        idx = close_all.index.get_indexer([t], method="pad")[0]
        if idx == -1:
            # no trading day before this week label
            continue

        t_real = close_all.index[idx]

        # Need enough history up to t_real
        if idx + 1 < min_history_days:
            continue

        # Need forward_days ahead
        if idx + forward_days >= len(close_all.index):
            continue

        # Slice window up to t_real (inclusive)
        window = df.iloc[: idx + 1].copy()
        close = close_all.iloc[: idx + 1]
        low = low_all.iloc[: idx + 1]

        # ----- Features -----
        ma20 = close.rolling(20).mean()
        ma20_last = float(ma20.iloc[-1])
        close_last = float(close.iloc[-1])

        if np.isnan(ma20_last) or ma20_last == 0:
            continue

        dist_ma20 = (close_last / ma20_last - 1.0) * 100.0

        # MA20 slope over last 5 MA20 points (regression)
        ma20_series = ma20.dropna()
        if len(ma20_series) < 6:
            continue
        y = ma20_series.iloc[-5:].to_numpy()
        x = np.arange(5)
        slope_5d = float(np.polyfit(x, y, 1)[0])

        # UDR10: % of up days in last 10 sessions
        if len(close) >= 11:
            last = close.iloc[-11:]
            udr10 = float((last.diff() > 0).sum()) / 10.0
        else:
            udr10 = np.nan

        # TPS10: % of last 10 days where Low >= MA20 (integrity)
        # Build aligned series (avoid rename pitfalls)
        valid = pd.concat([low, ma20], axis=1).dropna()
        valid.columns = ["Low", "MA20"]
        if len(valid) >= 10:
            recent = valid.tail(10)
            tps10 = float((recent["Low"] >= recent["MA20"]).mean())
        else:
            tps10 = np.nan

        # Vol ratio: std(returns 5d) / std(returns 20d)
        r = close.pct_change().dropna()
        if len(r) >= 21:
            v5 = float(r.tail(5).std())
            v20 = float(r.tail(20).std())
            vol_ratio = np.nan if v20 == 0 else v5 / v20
        else:
            vol_ratio = np.nan

        # Context: Pullback=1, Breakout=0, Other=-1 (using k=5 trading days)
        k = 5
        if len(close) >= (20 + k + 1):  # enough for MA20 and lookback
            close_k = float(close.iloc[-(k + 1)])
            ma_k = float(ma20.iloc[-(k + 1)])
            if (close_last > ma20_last) and (close_k <= ma_k):
                context = 0  # Breakout
            elif (close_last > ma20_last) and (close_k > ma_k) and (dist_ma20 <= 2.0):
                context = 1  # Pullback
            else:
                context = -1
        else:
            context = -1

        # ----- Labels (forward trading-day return) -----
        r5 = float(close_all.iloc[idx + forward_days] / close_all.iloc[idx] - 1.0)
        y_up = int(r5 >= 0.03)
        y_down = int(r5 <= -0.03)

        rows.append({
            "Ticker": ticker,
            "Date": pd.Timestamp(t_real),

            "Close": close_last,
            "Dist_to_MA20_%": dist_ma20,
            "MA20_slope_5d": slope_5d,
            "UDR10": udr10,
            "TPS10": tps10,
            "VolRatio_5_20": vol_ratio,
            "Context": context,

            "y_up": y_up,
            "y_down": y_down,
        })

    dprint(f"[{ticker}] rows generated={len(rows)}")
    return rows

def main():
    tickers = load_all_tickers()
    all_rows = []

    for i, t in enumerate(tickers, start=1):
        print(f"\nProcessing {i}/{len(tickers)}:", t)

        try:
            rows = build_rows_for_ticker(t, "5y")
            print(f"Rows from {t}:", len(rows))

            if not rows:
                log_bad(t, "no_rows_generated (no data or insufficient history)")
            else:
                all_rows.extend(rows)
        except Exception as e:
            log_bad(t, f"exception: {type(e).__name__}: {e}")
            continue

        # periodic checkpoint
        if i % 25 == 0 and all_rows:
            tmp = pd.DataFrame(all_rows)
            tmp.to_parquet(OUT_PATH, index=False)
            print(f"[checkpoint] tickers processed: {i}/{len(tickers)}, rows: {len(tmp)}")

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved dataset: {OUT_PATH}  rows={len(df)}  tickers={df['Ticker'].nunique() if not df.empty else 0}")

if __name__ == "__main__":
    rows = build_rows_for_ticker("AME", period="5y", debug=True)
    print("AME rows:", len(rows))
    main()
