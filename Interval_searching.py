import yfinance as yf
import pandas as pd
import numpy as np

def calculate_ATR(df, window):
    """
    Calculate the Average True Range (ATR) using a rolling mean of the True Range.
    By default, uses a 20-period ATR.
    """
    df = df.copy()

    # Shift the close by 1 day to get 'previous close'
    df['prev_close'] = df['Close'].shift(1)
    # Components of True Range (TR)
    df['high_low'] = df['High'] - df['Low']
    df.columns = df.columns.droplevel([1])
    df['high_pc'] = abs(df['High'] - df['prev_close'])
    df['low_pc'] = abs(df['Low'] - df['prev_close'])

    # True Range is the maximum of these three differences
    df['TR'] = df[['high_low', 'high_pc', 'low_pc']].max(axis=1)

    # Average True Range = rolling mean of the TR
    df[f'ATR_{window}'] = df['TR'].rolling(window).mean()
    return df

def find_ATR(symbol):
    interval_choice=["1d", "60m", "30m", "15m", "5m","2m"]
    window={"1d":10,"60m":80,"30m":80, "15m":80,"5m":80,"2m":160}
    for interval in interval_choice:
        df_daily=yf.download(symbol,period="1mo", interval=interval)
        df_daily = calculate_ATR(df_daily, window=window[interval])
        if df_daily[f'ATR_{window[interval]}'].iloc[-1]/df_daily['Close'].iloc[-1]<0.02:
            return interval, df_daily[f'ATR_{window[interval]}'].iloc[-1]

if __name__=='__main__':
    symbol = "KO"  # or any other ticker
    period_daily = "1y"  # daily data for 1 year
    interval_daily = "1d"

# Download daily data
    df_daily = yf.download("KO", period="1mo", interval="1d")
# Calculate 20-day ATR
    df_daily = calculate_ATR(df_daily, window=20)

# Get the latest close and ATR from the daily data
    latest_close_daily = df_daily['Close'].iloc[-1]
    latest_atr_daily   = df_daily['ATR_20'].iloc[-1]

    print(f"Latest daily close for {symbol}: {latest_close_daily:.2f}")
    print(f"Latest 20-day ATR for {symbol} (daily): {latest_atr_daily:.2f}")
    interval, ATR=find_ATR("KO")
    print(f"interval of {symbol} is {interval}, ATR is{ATR}")