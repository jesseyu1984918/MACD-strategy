import yfinance as yf
import pandas as pd
import numpy as np
import Interval_searching as Is
TRAILING_DAY=5
MACD_CHOICE='FAST'
MACD={'FAST':[6,19],'SLOW':[12,26]}
# Load NASDAQ and NYSE tickers from CSV files



# MACD Calculation Function
def calculate_macd(df, short_window=MACD[MACD_CHOICE][0], long_window=MACD[MACD_CHOICE][1], signal_window=9):
    df['EMA12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df


# Function to find stocks where MACD crosses above signal line from below, with MACD and signal both below zero within last 10 days
def find_macd_up_cross_below_zero(stock,interval):
    try:
        data = yf.download(stock, period="1mo", interval=interval)  # Fetch 1 month of daily data
        print(interval)
        data = calculate_macd(data)

        # Find cases where MACD crosses above Signal Line from below, with both MACD and Signal below zero
        data['MACD_Up_Cross'] = np.where(
            (data['MACD'] > data['Signal_Line']) &
            (data['MACD'].shift(1) < data['Signal_Line'].shift(1)) &
            (data['MACD'] < 0) &
            (data['Signal_Line'] < 0), True, False
        )
#just add some comments
        # Check for any upward cross meeting the criteria within the last 10 days
        recent_cross = data.tail(TRAILING_DAY)[data['MACD_Up_Cross']]

        # Return only if there's an upward cross within the last 10 days that meets all criteria
        return recent_cross[['Close', 'MACD', 'Signal_Line']] if not recent_cross.empty else None
    except Exception as e:
        print(f"Error processing {stock}: {e}")
        return None

def find_macd_down_cross_above_zero(stock,interval):
    try:
        data = yf.download(stock, period="3mo", interval=interval)  # Fetch 1 month of daily data
        data = calculate_macd(data)

        # Find cases where MACD crosses above Signal Line from below, with both MACD and Signal below zero
        data['MACD_Down_Cross'] = np.where(
            (data['MACD'] < data['Signal_Line']) &
            (data['MACD'].shift(1) > data['Signal_Line'].shift(1)) &
            (data['MACD'] > 0) &
            (data['Signal_Line'] > 0), True, False
        )

        # Check for any upward cross meeting the criteria within the last 10 days
        recent_cross = data.tail(TRAILING_DAY)[data['MACD_Down_Cross']]

        # Return only if there's an upward cross within the last 10 days that meets all criteria
        return recent_cross[['Close', 'MACD', 'Signal_Line']] if not recent_cross.empty else None
    except Exception as e:
        print(f"Error processing {stock}: {e}")
        return None
# Running the function for all symbols

def MACD_screening(all_tickers):
    up_results = []
    down_results = []
    for symbol in all_tickers:
        try:
            interval, ATR,trend,diff= Is.find_ATR(symbol)
        except:
            interval = "60m"
        cross_up_data = find_macd_up_cross_below_zero(symbol, interval)
        cross_down_data = find_macd_down_cross_above_zero(symbol, interval)
        if cross_up_data is not None:
            up_results.append({"symbol":symbol,"ATR interval":interval,"period average MA slope":trend, "close slope average":diff})
        if cross_down_data is not None:
            down_results.append({"symbol":symbol,"ATR interval":interval,"period average MA slope":trend, "close slope average":diff})
    return up_results,down_results
    # Display results


if __name__=="__main__":
    up_results = {}
    down_results = {}
    for symbol in all_tickers:
        try:
            interval, ATR,trend,diff=Is.find_ATR(symbol)
        except:
            interval="60m"
        cross_up_data = find_macd_up_cross_below_zero(symbol,interval)
        cross_down_data= find_macd_down_cross_above_zero(symbol,interval)
        if cross_up_data is not None:
            up_results[symbol] = interval
        if cross_down_data is not None:
            down_results[symbol]=interval

# Display results
    if up_results:
        for symbol,interval in up_results.items():
            print(
                f"MACD Upward Cross Above Signal Line with MACD and Signal Below Zero for {symbol} within last 10 period, interval is {interval}:\n")
    else:
        print(
            "No stocks found where MACD crosses above the signal line within the last 2 days with both MACD and signal below zero.")
    print( " Below is down crossing")
    if down_results:
        for symbol, df in down_results.items():
            print(
                f"MACD Down Cross Below Signal Line with MACD and Signal Above Zero for {symbol} within last 10 period, interval is {interval}:\n")
    else:
        print(
            "No stocks found where MACD crosses above the signal line within the last 2 days with both MACD and signal above zero.")
