import yfinance as yf
import pandas as pd
import numpy as np

# Load NASDAQ and NYSE tickers from CSV files
nasdaq_tickers = pd.read_csv('nasdaq-listed-symbols.csv')['Symbol'].tolist()
#nyse_tickers = pd.read_csv('nyse_tickers.csv')['Symbol'].tolist()
all_tickers = nasdaq_tickers  # Combine lists


# MACD Calculation Function
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['EMA12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df


# Function to find stocks where MACD crosses above signal line from below, with MACD and signal both below zero within last 10 days
def find_macd_up_cross_below_zero(stock):
    try:
        data = yf.download(stock, period="3mo", interval="1d")  # Fetch 1 month of daily data
        data = calculate_macd(data)

        # Find cases where MACD crosses above Signal Line from below, with both MACD and Signal below zero
        data['MACD_Up_Cross'] = np.where(
            (data['MACD'] > data['Signal_Line']) &
            (data['MACD'].shift(1) < data['Signal_Line'].shift(1)) &
            (data['MACD'] < 0) &
            (data['Signal_Line'] < 0), True, False
        )

        # Check for any upward cross meeting the criteria within the last 10 days
        recent_cross = data.tail(3)[data['MACD_Up_Cross']]

        # Return only if there's an upward cross within the last 10 days that meets all criteria
        return recent_cross[['Close', 'MACD', 'Signal_Line']] if not recent_cross.empty else None
    except Exception as e:
        print(f"Error processing {stock}: {e}")
        return None


# Running the function for all symbols
results = {}
for symbol in all_tickers:
    cross_data = find_macd_up_cross_below_zero(symbol)
    if cross_data is not None:
        results[symbol] = cross_data

# Display results
if results:
    for symbol, df in results.items():
        print(
            f"MACD Upward Cross Above Signal Line with MACD and Signal Below Zero for {symbol} within last 10 days:\n")
else:
    print(
        "No stocks found where MACD crosses above the signal line within the last 10 days with both MACD and signal below zero.")
