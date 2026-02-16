import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import date

CSV_PATH = "nasdaq_100_spy_500.csv"
LOOKBACK = "1y"

# ---------- helper ----------
def scalar(x):
    """Safely extract a float from pandas scalar or 1-element Series"""
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)

# -----------------------
# Load symbols
# -----------------------
symbols = pd.read_csv(CSV_PATH).iloc[:,0].dropna().astype(str).tolist()

closes=[]; highs=[]; lows=[]; vols=[]

for s in symbols:
    try:
        d = vbt.YFData.download(s, period=LOOKBACK)
        c = d.get("Close")
        if c.dropna().shape[0] < 200:
            continue

        closes.append(c.rename(s))
        highs.append(d.get("High").rename(s))
        lows.append(d.get("Low").rename(s))
        vols.append(d.get("Volume").rename(s))

        print("OK ", s)
    except:
        print("BAD", s)

# combine
close = pd.concat(closes, axis=1).sort_index().ffill()
high  = pd.concat(highs, axis=1).reindex_like(close).ffill()
low   = pd.concat(lows, axis=1).reindex_like(close).ffill()
vol   = pd.concat(vols, axis=1).reindex_like(close).ffill()

# remove accidental duplicate columns
close = close.loc[:,~close.columns.duplicated()]
high  = high.loc[:,~high.columns.duplicated()]
low   = low.loc[:,~low.columns.duplicated()]
vol   = vol.loc[:,~vol.columns.duplicated()]

# -----------------------
# Structure filters
# -----------------------
ma20  = close.rolling(20).mean()
ma50  = close.rolling(50).mean()
ma150 = close.rolling(150).mean()

trend = (close > ma50) & (ma50 > ma150)

hh50 = close.rolling(50).max()
recent_high = (high >= hh50.shift(1)).rolling(5).max()

not_extended = close <= ma20 * 1.10
pullback_zone = close >= ma20 * 0.95

candidate = trend & recent_high & not_extended & pullback_zone

# -----------------------
# ORIGINAL SCORING
# -----------------------
distance_from_high = (hh50 - close) / hh50
trend_strength = 1 - distance_from_high

tr = high - low
atr20 = tr.rolling(20).mean()
atr200 = tr.rolling(200).mean()
tightness = (1 - atr20 / atr200).clip(0,1)

pullback_quality = (close - ma20) / ma20
pullback_quality = 1 - pullback_quality.abs().clip(upper=0.1) / 0.1

score = 0.5*trend_strength + 0.3*tightness + 0.2*pullback_quality

today_score = score.iloc[-1].where(candidate.iloc[-1]).dropna().sort_values(ascending=False)

# -----------------------
# Output
# -----------------------
print("\nDate:", date.today())
print("Ranked candidates:", len(today_score))
print("")

last_date = close.index[-1]

for i, (s, val) in enumerate(today_score.head(20).items(), 1):
    val = scalar(val)
    dist = scalar(close.loc[last_date, s] / ma20.loc[last_date, s] - 1) * 100
    print(f"{i:2d}. {s:6s}  score:{val:.3f}  dist_from_MA20:{dist:5.2f}%")
