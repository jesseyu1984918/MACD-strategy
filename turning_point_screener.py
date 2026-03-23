import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
import sys
from trade_guardrails import (
    MIN_AVG_DOLLAR_VOL,
    MIN_PRICE,
    MAX_10D_RUNUP_PCT,
    MAX_ABOVE_MA20_PCT,
    build_metadata_flags,
    flatten_columns,
)
# =====================================================
# SETTINGS
# =====================================================
INPUT_CSV = sys.argv[1]
PERIOD = "2y"
INTERVAL = "1d"

DAILY_MA = 20
ATR_WIN = 14
VOL_WIN = 20
NEAR_MA_PCT = 0.05

OUTPUT_FILE = "ma_turning_rank_vectorbt.csv"


# =====================================================
# LOAD TICKERS
# =====================================================
tickers = pd.read_csv(INPUT_CSV)["Symbol"].dropna().astype(str).unique().tolist()
leveraged_flags = build_metadata_flags(tickers)
print(f"Tickers requested: {len(tickers)}")


# =====================================================
# SAFE DOWNLOADER
# =====================================================
def download_data(tickers):
    close_dict, high_dict, low_dict, vol_dict = {}, {}, {}, {}

    for t in tickers:
        try:
            df = yf.download(t, period=PERIOD, interval=INTERVAL, progress=False)
            df = flatten_columns(df)

            if df is None or len(df) < 120:
                print(f"Skipping {t} (not enough data)")
                continue

            df.index = pd.to_datetime(df.index)

            close_dict[t] = df["Close"]
            high_dict[t] = df["High"]
            low_dict[t] = df["Low"]
            vol_dict[t] = df["Volume"]

            print(f"Loaded {t}")

        except Exception as e:
            print(f"Failed {t}: {e}")

    close = pd.DataFrame(close_dict).sort_index()
    high = pd.DataFrame(high_dict).reindex(close.index)
    low = pd.DataFrame(low_dict).reindex(close.index)
    vol = pd.DataFrame(vol_dict).reindex(close.index)

    return close.astype(float), high.astype(float), low.astype(float), vol.astype(float)


close, high, low, volume = download_data(tickers)

if len(close.columns) == 0:
    raise SystemExit("No valid tickers downloaded")

print(f"Valid tickers after download: {len(close.columns)}")


# =====================================================
# DAILY FEATURES (vectorbt MA, pandas math)
# =====================================================
ma_vbt = vbt.MA.run(close, window=DAILY_MA).ma

# remove vectorbt metadata → avoid pandas alignment crash
ma = pd.DataFrame(ma_vbt.values, index=close.index, columns=close.columns)

atr = (high - low).rolling(ATR_WIN).mean()

slope = (ma - ma.shift(1)) / atr
curvature = slope - slope.shift(1)

dist = (close - ma) / ma
vol = close.pct_change().rolling(VOL_WIN).std()
runup_10d_pct = close.pct_change(10) * 100
avg_dollar_vol_20d = (close * volume).rolling(20).mean()
price_ok = close >= MIN_PRICE
liquidity_ok = avg_dollar_vol_20d >= MIN_AVG_DOLLAR_VOL
runup_ok = runup_10d_pct <= MAX_10D_RUNUP_PCT
ma_extension_ok = dist <= (MAX_ABOVE_MA20_PCT / 100)
non_leveraged = pd.DataFrame(
    {col: not bool(leveraged_flags.get(col, False)) for col in close.columns},
    index=close.index,
)

turning = (curvature > 0) & (curvature.shift(1) <= 0)


# =====================================================
# WEEKLY TREND FILTER
# =====================================================
weekly_close = close.resample("W-FRI").last()
weekly_ma = weekly_close.rolling(20).mean()
weekly_slope = weekly_ma - weekly_ma.shift(1)

weekly_slope = weekly_slope.dropna(how="all")

if len(weekly_slope) == 0:
    raise SystemExit("Not enough weekly data")

weekly_last = weekly_slope.iloc[-1]

weekly_trend = pd.Series(index=weekly_last.index, dtype="object")
weekly_trend[weekly_last > 0] = "UP"
weekly_trend[(weekly_last <= 0) & (weekly_last > -0.001)] = "FLAT"
weekly_trend[weekly_last <= -0.001] = "DOWN"

weekly_trend = weekly_trend.dropna()

# keep only valid tickers
valid_cols = weekly_trend.index
close = close[valid_cols]
ma = ma[valid_cols]
slope = slope[valid_cols]
curvature = curvature[valid_cols]
dist = dist[valid_cols]
vol = vol[valid_cols]
turning = turning[valid_cols]
runup_10d_pct = runup_10d_pct[valid_cols]
avg_dollar_vol_20d = avg_dollar_vol_20d[valid_cols]
price_ok = price_ok[valid_cols]
liquidity_ok = liquidity_ok[valid_cols]
runup_ok = runup_ok[valid_cols]
ma_extension_ok = ma_extension_ok[valid_cols]
non_leveraged = non_leveraged[valid_cols]


# =====================================================
# LAST VALUES
# =====================================================
last_price = close.iloc[-1]
last_slope = slope.iloc[-1]
last_curv = curvature.iloc[-1]
last_dist = dist.iloc[-1]
last_vol = vol.iloc[-1]
last_turn = turning.iloc[-1]
last_runup_10d = runup_10d_pct.iloc[-1]
last_avg_dollar_vol_20d = avg_dollar_vol_20d.iloc[-1]
last_guardrail = (price_ok & liquidity_ok & runup_ok & ma_extension_ok & non_leveraged).iloc[-1]


# =====================================================
# SCORE MODEL
# =====================================================
curvature_score = np.tanh(last_curv * 6)
slope_score = np.tanh((last_slope + 0.05) * 5)
dist_score = 1 - np.minimum(np.abs(last_dist) / NEAR_MA_PCT, 1)
vol_score = np.tanh((-last_vol) * 20)

trend_bonus = weekly_trend.map({"UP": 20, "FLAT": 5, "DOWN": -40})

score = (
    40 * curvature_score +
    25 * slope_score +
    20 * dist_score +
    15 * vol_score +
    trend_bonus
).round(2)


# =====================================================
# OUTPUT
# =====================================================
results = pd.DataFrame({
    "Ticker": score.index,
    "Score": score.values,
    "WeeklyTrend": weekly_trend.values,
    "TurningToday": last_turn.values,
    "GuardrailsPassed": last_guardrail.values,
    "Price": np.round(last_price.values, 2),
    "Slope": np.round(last_slope.values, 4),
    "Curvature": np.round(last_curv.values, 4),
    "Dist_from_MA_%": np.round(last_dist.values * 100, 2),
    "DailyVol_%": np.round(last_vol.values * 100, 2),
    "Runup_10d_%": np.round(last_runup_10d.values, 2),
    "AvgDollarVol20d_M": np.round(last_avg_dollar_vol_20d.values / 1_000_000, 2),
})

results = results[results["WeeklyTrend"] != "DOWN"]
results = results[results["GuardrailsPassed"]]
results = results.sort_values("Score", ascending=False)

print("\n=== TOP TURNING CANDIDATES ===\n")
print(results.head(20).to_string(index=False))

results.to_csv("ma_turning_rank_vectorbt.csv", index=False)
print("\nSaved to ma_turning_rank_vectorbt.csv")
