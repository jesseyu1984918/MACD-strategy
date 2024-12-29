import streamlit as st
import Interval_searching as Is
import pandas as pd
import MACD_screening as MACD
import numpy as np
def main():
    st.title("Stock Pick Panel")
    st.write("Use the sidebar to select which analysis you want to run.")

    # 1) Sidebar radio to switch between MACD screening and ATR finder
    analysis_choice = st.sidebar.radio(
        "Choose an analysis function:",
        ("MACD Screening", "ATR Finder")
    )

    # 2) User input for tickers
    tickers_input = st.text_input(
        "Enter ticker symbols (comma-separated):",
        value="AAPL, TSLA, MSFT"
    )
    # Clean up the ticker list
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if analysis_choice == "MACD Screening":
        st.subheader("MACD Screening")
        st.write("Screen selected tickers using your MACD logic.")
        # 1) CSV file uploader
        csv_file = st.file_uploader("Upload your CSV file", type=["csv"])

        # If a file is uploaded, process it
        if csv_file is not None:
            df = pd.read_csv(csv_file)

            # Ensure 'symbol' column exists
            if "Symbol" not in df.columns:
                st.error("The uploaded CSV must have a 'symbol' column.")
            else:
                # Extract tickers from the 'symbol' column
                tickers = df["Symbol"].tolist()

        if st.button("Run MACD Screening"):
            up_results,down_results = MACD.MACD_screening(tickers)
            st.write("MACD up crossing")
            st.dataframe(up_results)
            st.write("MACD down crossing")
            st.dataframe(down_results)
    elif analysis_choice == "ATR Finder":
        st.subheader("ATR Finder")
        st.write("Find the ATR (Average True Range) of selected tickers.")

        if st.button("Run ATR Finder"):
            results,ATR,trend,diff = Is.find_ATR(tickers)
            st.dataframe({"Interval":results,"ATR":ATR,"period EMA average slope":trend,"period average close diff":diff})


if __name__ == "__main__":
    main()