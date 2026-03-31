from __future__ import annotations

from io import StringIO
from tempfile import NamedTemporaryFile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import Interval_searching as Is
import MACD_screening as MACD
import macro_market_status
import news_sentiment
import position_exit_review
import scanner
import system_signal_backtest
from trade_guardrails import build_metadata_flags


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
    st.write("Rank fresh setups and breakouts, store a dated snapshot, and show how ranks moved versus the previous run.")
    st.markdown(
        """
        **Principles**

        - The scanner is designed for long ideas that are already in trend, liquid enough to trade, and not overly extended.
        - It looks for two states: `SETUP` names sitting close to a breakout pivot, and `BREAKOUT` names that cleared the pivot recently and are still controlled.
        - It now runs in a higher-conviction mode. Besides the normal trend, liquidity, and extension checks, names must also show 20-day relative strength versus `SPY` and `QQQ`.
        - It blocks names that are too illiquid, too cheap, too hot, too stretched from the 20-day average, too far below the 20-day average, structurally out of trend, or too weak versus the major indexes.
        - Macro market status is now part of the process. `Risk-on` slightly boosts scanner scores, `Risk-off` suppresses them, so the list is more regime-aware.
        """
    )
    st.markdown(
        """
        **Weighted Score Guide**

        `Score` is the weighted scanner score, or `WScore`. Higher is better, but it is a ranking tool, not a standalone buy signal.

        - `0.60+`: stronger quality. Trend, tightness, and location are generally aligned.
        - `0.40` to `0.60`: usable but more mixed. Usually needs cleaner price action or better market support.
        - Below `0.40`: weak quality. Treat as lower-priority unless there is a very specific reason.

        What drives the score:

        - proximity to the breakout pivot
        - price tightness versus longer-term ATR
        - trend strength from the 50-day versus 150-day averages
        - how stretched price is from the 20-day average
        - how far price is above the breakout level
        - breakout freshness
        - whether the name is still a `SETUP` or already a `BREAKOUT`
        - current macro regime

        What is not inside `WScore` but is still required:

        - at least about `+5%` relative strength versus `SPY` over 20 trading days
        - non-negative relative strength versus `QQQ` over 20 trading days
        """
    )

    if st.button("Run Ranked Scanner"):
        ranked_df, blocked_df = scanner.run_scanner(symbols)
        ranked_df, blocked_df = scanner.persist_rank_history(ranked_df, blocked_df, source_label="stock_center")

        st.write(f"Date: {pd.Timestamp.now().date()}")
        st.write(f"Ranked candidates: {len(ranked_df)}")
        st.write("Top ranked candidates")
        st.dataframe(ranked_df)

        if not ranked_df.empty:
            st.write(f"Rank history saved to {scanner.RANK_HISTORY_OUTPUT}")

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


def render_one_stock_analysis() -> None:
    st.subheader("One Stock Analysis")
    st.write("Run the ranked scanner logic on a single ticker, show the current result, and back-calculate its historical WScore over time.")

    symbol = st.text_input("Enter one ticker:", value="AAPL", key="single_stock_symbol").strip().upper()

    if st.button("Run One Stock Analysis"):
        if not symbol:
            st.error("Enter a ticker first.")
            return

        ranked_df, blocked_df = scanner.run_scanner([symbol])

        if not ranked_df.empty:
            st.success(f"{symbol} passed the screen.")
            st.dataframe(ranked_df)
            row = ranked_df.iloc[0]
            st.metric("WScore", f"{row['Score']:.3f}")
            st.caption(
                f"Signal: {row['SignalType']} | Trade action: {row['TradeAction']} | "
                f"Macro regime: {row['MacroRegime']}"
            )
            return

        st.warning(f"{symbol} did not pass the screen.")
        if not blocked_df.empty:
            st.dataframe(blocked_df)
            blocked_row = blocked_df[blocked_df["Symbol"] == symbol]
            if not blocked_row.empty:
                st.caption(f"Blocked reason: {blocked_row.iloc[0]['blocked_reason']}")
        else:
            st.info("No ranked output and no blocked output were returned.")

    st.write("Back-calculated WScore history")
    hist = scanner.download_symbol(symbol, lookback="2y")
    if hist is None or hist.empty:
        st.info(f"Could not load enough price history for {symbol}.")
        return

    close = pd.DataFrame({symbol: hist["Close"].rename(symbol)}).sort_index()
    high = pd.DataFrame({symbol: hist["High"].rename(symbol)}).reindex(close.index).ffill()
    low = pd.DataFrame({symbol: hist["Low"].rename(symbol)}).reindex(close.index).ffill()
    volume = pd.DataFrame({symbol: hist["Volume"].rename(symbol)}).reindex(close.index).ffill()
    leveraged_flags = build_metadata_flags([symbol])
    macro_state = system_signal_backtest.compute_macro_state(close.index)
    state = system_signal_backtest.compute_scanner_state(close, high, low, volume, leveraged_flags, macro_state)

    history_df = pd.DataFrame(
        {
            "Date": close.index,
            "Close": close[symbol],
            "WScore": state["raw_score"][symbol],
            "SignalType": pd.Series(index=close.index, dtype="object"),
            "Candidate": state["candidate"][symbol],
            "MacroRegime": macro_state["MacroRegime"].values,
        }
    )
    history_df.loc[state["setup_candidate"][symbol].fillna(False), "SignalType"] = "SETUP"
    history_df.loc[state["breakout_candidate"][symbol].fillna(False), "SignalType"] = "BREAKOUT"
    cutoff_date = pd.Timestamp.now().normalize() - pd.DateOffset(months=6)
    history_df = history_df[history_df["Date"] >= cutoff_date].copy()
    history_df = history_df.reset_index(drop=True)

    if history_df.empty:
        st.info(f"No back-calculated history was available for {symbol} during the last six months.")
        return

    exit_state = system_signal_backtest.compute_exit_state(close, volume)
    exit_weights: list[float] = []
    exit_recommendations: list[str] = []
    for row in history_df.itertuples(index=False):
        current_date = pd.Timestamp(row.Date)
        review_row = pd.Series(
            {
                "Side": "LONG",
                "LastPrice": float(row.Close),
                "MA20": exit_state["ma20"].at[current_date, symbol],
                "MA50": exit_state["ma50"].at[current_date, symbol],
                "MA150": exit_state["ma150"].at[current_date, symbol],
                "PnLPct": np.nan,
                "Runup10dPct": exit_state["runup_10d_pct"].at[current_date, symbol],
                "AvgDollarVol20dM": (
                    exit_state["avg_dollar_vol_20d"].at[current_date, symbol] / 1_000_000
                    if pd.notna(exit_state["avg_dollar_vol_20d"].at[current_date, symbol])
                    else np.nan
                ),
                "MACDHist": exit_state["macd_hist"].at[current_date, symbol],
                "MA20Slope": exit_state["ma20_slope"].at[current_date, symbol],
                "BreakoutAge": state["breakout_age"].at[current_date, symbol],
                "DistFromMA20Pct": state["extension_from_ma20"].at[current_date, symbol] * 100,
                "DistFromBreakoutPct": state["extension_from_breakout"].at[current_date, symbol] * 100,
                "ScannerScore": state["raw_score"].at[current_date, symbol],
                "ScannerTrend": state["trend"].at[current_date, symbol],
                "ScannerSignalType": row.SignalType if pd.notna(row.SignalType) else "",
            }
        )
        macro_snapshot = {
            "regime": macro_state.loc[current_date, "MacroRegime"],
            "score": macro_state.loc[current_date, "MacroScore"],
            "scanner_multiplier": macro_state.loc[current_date, "ScannerMultiplier"],
            "exit_review_threshold_shift": macro_state.loc[current_date, "ExitReviewThresholdShift"],
        }
        recommendation, _ = position_exit_review.build_recommendation(
            review_row,
            bool(leveraged_flags.get(symbol, False)),
            macro_snapshot,
        )
        exit_pressure = position_exit_review.build_exit_pressure(
            review_row,
            bool(leveraged_flags.get(symbol, False)),
            macro_snapshot,
        )
        exit_recommendations.append(recommendation)
        exit_weights.append(exit_pressure)

    history_df["ExitRecommendation"] = exit_recommendations
    history_df["ExitWeight"] = exit_weights

    score_line = (
        alt.Chart(history_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("WScore:Q", title="WScore"),
            color=alt.value("#1f77b4"),
            tooltip=["Date:T", "Close:Q", "WScore:Q", "SignalType:N", "Candidate:N", "MacroRegime:N"],
        )
        .properties(height=300)
    )
    candidate_points = (
        alt.Chart(history_df[history_df["Candidate"].fillna(False)])
        .mark_circle(size=70)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("WScore:Q", title="WScore"),
            color=alt.Color("SignalType:N", title="Signal"),
            tooltip=["Date:T", "Close:Q", "WScore:Q", "SignalType:N", "MacroRegime:N"],
        )
    )
    st.altair_chart(score_line + candidate_points, use_container_width=True)

    price_chart = (
        alt.Chart(history_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(
                "Close:Q",
                title="Close",
                scale=alt.Scale(
                    domain=[
                        float(history_df["Close"].min()) - max((float(history_df["Close"].max()) - float(history_df["Close"].min())) * 0.05, 0.5),
                        float(history_df["Close"].max()) + max((float(history_df["Close"].max()) - float(history_df["Close"].min())) * 0.05, 0.5),
                    ]
                ),
            ),
            color=alt.value("#2E6F40"),
            tooltip=["Date:T", "Close:Q", "WScore:Q", "SignalType:N", "MacroRegime:N"],
        )
        .properties(height=260)
    )
    st.altair_chart(price_chart, use_container_width=True)

    exit_chart = (
        alt.Chart(history_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(
                "ExitWeight:Q",
                title="Exit Weight",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("ExitRecommendation:N", title="Exit View"),
            tooltip=["Date:T", "ExitWeight:Q", "ExitRecommendation:N", "WScore:Q", "Close:Q"],
        )
        .properties(height=260)
    )
    st.altair_chart(exit_chart, use_container_width=True)
    st.dataframe(history_df.sort_values("Date", ascending=False))


def render_position_exit_review() -> None:
    st.subheader("Position Exit Review")
    st.write("Upload your transaction history export and get HOLD / REVIEW / EXIT suggestions for open positions.")
    txn_file = st.file_uploader("Upload transaction history CSV", type=["csv"], key="txn_history")

    if st.button("Run Position Exit Review"):
        if txn_file is None:
            st.error("Upload a transaction history CSV first.")
            return

        with NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(txn_file.getvalue())
            tmp_path = tmp_file.name

        review_df = position_exit_review.review_positions_from_csv(tmp_path)
        st.write(f"Open positions reviewed: {len(review_df)}")
        st.dataframe(review_df)

        if not review_df.empty:
            csv_bytes = review_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download position review CSV",
                data=csv_bytes,
                file_name="position_exit_review.csv",
                mime="text/csv",
            )


def render_news_sentiment(symbols: list[str]) -> None:
    st.subheader("News Sentiment")
    st.write("Fetch recent Yahoo Finance headlines and score each symbol from 0 to 10.")

    if st.button("Run News Sentiment"):
        news_df = news_sentiment.get_symbol_news_sentiment_df(symbols)
        st.dataframe(news_df)

        if not news_df.empty:
            csv_bytes = news_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download news sentiment CSV",
                data=csv_bytes,
                file_name="news_sentiment.csv",
                mime="text/csv",
            )


def render_macro_market_status() -> None:
    st.subheader("Macro Market Status")
    st.write("Check the broad market trend, volatility backdrop, and risk appetite before acting on individual setups.")

    if st.button("Run Macro Market Status"):
        macro_df = macro_market_status.build_macro_status_df()
        macro_trend_df = macro_market_status.build_macro_trend_df()
        summary = macro_market_status.build_macro_summary(macro_df)

        col1, col2, col3 = st.columns(3)
        col1.caption("Market Regime")
        col1.write(summary["regime"])
        col2.caption("Trend")
        col2.write(summary["trend"])
        col3.caption("Risk")
        col3.write(summary["risk"])

        if not macro_trend_df.empty:
            st.write("Trend view")
            chart_columns = st.columns(2)
            for idx, column_name in enumerate(macro_trend_df.columns):
                series_df = macro_trend_df[[column_name]].dropna().reset_index()
                series_df.columns = ["Date", "Value"]
                y_min = float(series_df["Value"].min())
                y_max = float(series_df["Value"].max())
                padding = max((y_max - y_min) * 0.08, max(abs(y_max) * 0.01, 0.01))
                chart = (
                    alt.Chart(series_df)
                    .mark_line()
                    .encode(
                        x=alt.X("Date:T", title=None),
                        y=alt.Y(
                            "Value:Q",
                            title=None,
                            scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
                        ),
                    )
                    .properties(height=180)
                )
                chart_columns[idx % 2].caption(column_name)
                chart_columns[idx % 2].altair_chart(chart, use_container_width=True)

        st.write("Macro dashboard")
        st.dataframe(macro_df)

        if not macro_df.empty:
            csv_bytes = macro_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download macro market status CSV",
                data=csv_bytes,
                file_name="macro_market_status.csv",
                mime="text/csv",
            )


def render_ranking_history() -> None:
    st.subheader("Ranking History")
    st.write("Review timestamped ranking snapshots and see how each symbol moved over time.")

    if not pd.io.common.file_exists(scanner.RANK_HISTORY_OUTPUT):
        st.info("No ranking history yet. Run the Ranked Scanner first.")
        return

    history_df = pd.read_csv(scanner.RANK_HISTORY_OUTPUT)
    if history_df.empty:
        st.info("Ranking history file is empty.")
        return

    history_df["RunTimestamp"] = pd.to_datetime(history_df["RunTimestamp"], errors="coerce")
    symbols = sorted(history_df["Symbol"].dropna().unique().tolist())
    selected_symbol = st.selectbox("Filter by symbol", options=["All"] + symbols, index=0)

    filtered_df = history_df if selected_symbol == "All" else history_df[history_df["Symbol"] == selected_symbol].copy()
    filtered_df = filtered_df.sort_values(["RunTimestamp", "Rank"])

    st.write(f"Snapshots loaded: {filtered_df['RunTimestamp'].nunique()}")
    st.dataframe(filtered_df)

    if selected_symbol != "All":
        st.line_chart(filtered_df.set_index("RunTimestamp")["Rank"])

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download ranking history CSV",
        data=csv_bytes,
        file_name="scanner_rank_history_filtered.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Stock Pick Panel")
    st.write("Use the sidebar to select which analysis you want to run.")

    analysis_choice = st.sidebar.radio(
        "Choose an analysis function:",
        (
            "Ranked Scanner",
            "One Stock Analysis",
            "Ranking History",
            "Macro Market Status",
            "MACD Screening",
            "ATR Finder",
            "Position Exit Review",
            "News Sentiment",
        ),
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
    elif analysis_choice == "One Stock Analysis":
        render_one_stock_analysis()
    elif analysis_choice == "Ranking History":
        render_ranking_history()
    elif analysis_choice == "Macro Market Status":
        render_macro_market_status()
    elif analysis_choice == "MACD Screening":
        render_macd_screening(symbols)
    elif analysis_choice == "ATR Finder":
        render_atr_finder(symbols)
    elif analysis_choice == "Position Exit Review":
        render_position_exit_review()
    else:
        render_news_sentiment(symbols)


if __name__ == "__main__":
    main()
