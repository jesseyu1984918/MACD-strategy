from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

import Interval_searching as Is
import MACD_screening as MACD
import scanner


def parse_uploaded_symbols(csv_file) -> list[str]:
    raw_text = csv_file.getvalue().decode("utf-8")
    raw = pd.read_csv(StringIO(raw_text), header=None)
    first_value = str(raw.iloc[0, 0]).strip().lower() if not raw.empty else ""

    if first_value in {"symbol", "symbols", "ticker", "tickers"}:
        raw = pd.read_csv(StringIO(raw_text))

    symbols = raw.iloc[:, 0].dropna().astype(str).str.strip()
    symbols = symbols[symbols.str.lower().ne("symbol")]
    return symbols.tolist()


def load_symbols_from_inputs(tickers_input: str, csv_file) -> list[str]:
    if csv_file is not None:
        return parse_uploaded_symbols(csv_file)
    return [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]


def render_macd_screening(symbols: list[str]) -> None:
    st.subheader("MACD Screening")
    st.write("Screen selected tickers using your MACD logic.")

    if st.button("Run MACD Screening"):
        up_results, down_results = MACD.MACD_screening(symbols)
        st.write("MACD up crossing")
        st.dataframe(pd.DataFrame(up_results))
        st.write("MACD down crossing")
        st.dataframe(pd.DataFrame(down_results))


def render_atr_finder(symbols: list[str]) -> None:
    st.subheader("ATR Finder")
    st.write("Find the ATR (Average True Range) of selected tickers.")

    if st.button("Run ATR Finder"):
        rows = []
        for symbol in symbols:
            try:
                interval, atr, trend, diff = Is.find_ATR(symbol)
                rows.append(
                    {
                        "Symbol": symbol,
                        "Interval": interval,
                        "ATR": atr,
                        "Period MA Average Slope": trend,
                        "Period Average Close Diff": diff,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "Symbol": symbol,
                        "Interval": None,
                        "ATR": None,
                        "Period MA Average Slope": None,
                        "Period Average Close Diff": None,
                        "Error": str(exc),
                    }
                )
        st.dataframe(pd.DataFrame(rows))


def render_ranked_scanner(symbols: list[str]) -> None:
    st.subheader("Ranked Scanner")
    st.write("Rank fresh setups and breakouts, then show exactly why other symbols were blocked.")

    if st.button("Run Ranked Scanner"):
        ranked_df, blocked_df = scanner.run_scanner(symbols)

        st.write(f"Date: {pd.Timestamp.now().date()}")
        st.write(f"Ranked candidates: {len(ranked_df)}")
        st.write("Top ranked candidates")
        st.dataframe(ranked_df)

        st.write("Blocked symbols")
        st.dataframe(blocked_df)

        if not blocked_df.empty:
            csv_bytes = blocked_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download blocked symbols CSV",
                data=csv_bytes,
                file_name="scanner_blocked.csv",
                mime="text/csv",
            )


def main() -> None:
    st.title("Stock Pick Panel")
    st.write("Use the sidebar to select which analysis you want to run.")

    analysis_choice = st.sidebar.radio(
        "Choose an analysis function:",
        ("Ranked Scanner", "MACD Screening", "ATR Finder"),
    )

    tickers_input = st.text_input(
        "Enter ticker symbols (comma-separated):",
        value="AAPL, TSLA, MSFT",
    )
    csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
    symbols = load_symbols_from_inputs(tickers_input, csv_file)

    st.caption(f"Symbols loaded: {len(symbols)}")

    if analysis_choice == "Ranked Scanner":
        render_ranked_scanner(symbols)
    elif analysis_choice == "MACD Screening":
        render_macd_screening(symbols)
    else:
        render_atr_finder(symbols)


if __name__ == "__main__":
    main()
