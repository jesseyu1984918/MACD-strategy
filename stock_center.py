from __future__ import annotations

from io import StringIO
from pathlib import Path
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


STRATEGY_PRESETS = {
    "classic": {
        "label": "Full Multi-Signal Scanner",
        "description": "Default participation mode: rank setups, breakouts, reclaims, momentum, and continuing leaders.",
    },
    "best_breakout": {
        "label": "Selective Breakout Only",
        "description": "Narrow optional mode: fresh breakouts only, score >= 0.60, rank <= 10, RS >= 8 vs SPY and >= 2 vs QQQ. It will intentionally miss many rising stocks.",
    },
}

APP_ROOT = Path(__file__).resolve().parent
DATASET_DIR = APP_ROOT / "dataset"
MAJOR_INDEX_ETF_FILE = DATASET_DIR / "major_index_etf.csv"
MAJOR_INDEX_ETF_DEFAULTS = {
    "trend_mode": "ma20",
    "entry_mode": "all_candidates",
    "min_entry_score": 0.55,
    "max_entry_rank": 10,
    "min_rs_vs_spy_20d": 0.0,
    "min_rs_vs_qqq_20d": -2.0,
}
MAJOR_INDEX_ETF_EXIT_POLICY = {
    "recommended": "exit_only with 4 confirmation days",
    "alternate": "exit_only with 3 confirmation days plus 10%/5% profit lock",
    "exit_on": "exit_only",
    "exit_confirm_days": 4,
    "profit_lock_trigger_pct": 0.0,
    "profit_lock_drawdown_pct": 0.0,
}
MAJOR_INDEX_BACKTEST_VARIANTS = [
    {
        "name": "ma20_breakout_rank10_rs50",
        "label": "MA20 breakout rank<=10 RS 5/0",
        "trend_mode": "ma20",
        "entry_mode": "breakout_only",
        "min_entry_score": 0.60,
        "max_entry_rank": 10,
        "max_new_positions_per_day": 4,
        "exit_confirm_days": 3,
        "min_rs_vs_spy_20d": 5.0,
        "min_rs_vs_qqq_20d": 0.0,
    },
    {
        "name": "ma20_breakout_rank10_rs0m2",
        "label": "MA20 breakout rank<=10 RS 0/-2",
        "trend_mode": "ma20",
        "entry_mode": "breakout_only",
        "min_entry_score": 0.60,
        "max_entry_rank": 10,
        "max_new_positions_per_day": 4,
        "exit_confirm_days": 3,
        "min_rs_vs_spy_20d": 0.0,
        "min_rs_vs_qqq_20d": -2.0,
    },
    {
        "name": "ma20_all_rank10_rs0m2",
        "label": "MA20 all-candidates rank<=10 RS 0/-2",
        "trend_mode": "ma20",
        "entry_mode": "all_candidates",
        "min_entry_score": 0.55,
        "max_entry_rank": 10,
        "max_new_positions_per_day": 4,
        "exit_confirm_days": 3,
        "min_rs_vs_spy_20d": 0.0,
        "min_rs_vs_qqq_20d": -2.0,
    },
    {
        "name": "hybrid_breakout_rank10_rs50",
        "label": "Hybrid breakout rank<=10 RS 5/0",
        "trend_mode": "hybrid",
        "entry_mode": "breakout_only",
        "min_entry_score": 0.60,
        "max_entry_rank": 10,
        "max_new_positions_per_day": 4,
        "exit_confirm_days": 3,
        "min_rs_vs_spy_20d": 5.0,
        "min_rs_vs_qqq_20d": 0.0,
    },
    {
        "name": "ma50_breakout_rank4_rs50",
        "label": "MA50 breakout rank<=4 RS 5/0",
        "trend_mode": "ma50",
        "entry_mode": "breakout_only",
        "min_entry_score": 0.60,
        "max_entry_rank": 4,
        "max_new_positions_per_day": 4,
        "exit_confirm_days": 3,
        "min_rs_vs_spy_20d": 5.0,
        "min_rs_vs_qqq_20d": 0.0,
    },
]


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


def resolve_trend_mode(trend_mode: str, strategy_preset: str) -> str:
    if strategy_preset == "best_breakout":
        return "ma20"
    return trend_mode


def apply_strategy_preset_to_ranked_df(ranked_df: pd.DataFrame, strategy_preset: str) -> pd.DataFrame:
    if ranked_df.empty or strategy_preset == "classic":
        return ranked_df

    filtered = ranked_df[
        (ranked_df["SignalType"] == "BREAKOUT")
        & (ranked_df["Score"] >= 0.60)
        & (ranked_df["RSvsSPY20d"] >= 8.0)
        & (ranked_df["RSvsQQQ20d"] >= 2.0)
    ].copy()
    filtered = filtered.sort_values(["Score", "RSvsSPY20d", "RSvsQQQ20d"], ascending=[False, False, False]).head(10)
    filtered = filtered.reset_index(drop=True)
    return filtered


def load_major_index_etf_universe() -> pd.DataFrame:
    if not MAJOR_INDEX_ETF_FILE.exists():
        raise FileNotFoundError(f"Missing major ETF universe file: {MAJOR_INDEX_ETF_FILE}")
    universe = pd.read_csv(MAJOR_INDEX_ETF_FILE)
    universe.columns = [str(col).strip() for col in universe.columns]
    universe["Symbol"] = universe["Symbol"].astype(str).str.strip().str.upper()
    universe["Description"] = universe["Description"].fillna("").astype(str)
    universe = universe[universe["Symbol"].ne("")].drop_duplicates("Symbol").reset_index(drop=True)
    return universe


def get_major_index_etf_symbols() -> set[str]:
    try:
        return set(load_major_index_etf_universe()["Symbol"].tolist())
    except FileNotFoundError:
        return set()


def apply_major_index_etf_filter(ranked_df: pd.DataFrame) -> pd.DataFrame:
    if ranked_df.empty:
        return ranked_df
    ranked_df = scanner.add_rank_columns(ranked_df)
    filtered = ranked_df[
        (ranked_df["Score"] >= float(MAJOR_INDEX_ETF_DEFAULTS["min_entry_score"]))
        & (ranked_df["Rank"] <= int(MAJOR_INDEX_ETF_DEFAULTS["max_entry_rank"]))
        & (ranked_df["RSvsSPY20d"] >= float(MAJOR_INDEX_ETF_DEFAULTS["min_rs_vs_spy_20d"]))
        & (ranked_df["RSvsQQQ20d"] >= float(MAJOR_INDEX_ETF_DEFAULTS["min_rs_vs_qqq_20d"]))
    ].copy()
    return filtered.sort_values(["Score", "RSvsSPY20d", "RSvsQQQ20d"], ascending=[False, False, False]).reset_index(drop=True)


def annualize_trade_return(avg_return_pct: float, avg_hold_days: float) -> float | None:
    if pd.isna(avg_return_pct) or pd.isna(avg_hold_days) or avg_hold_days <= 0 or avg_return_pct <= -100:
        return None
    return (((1 + (avg_return_pct / 100.0)) ** (252.0 / avg_hold_days)) - 1) * 100.0


def summarize_backtest_variant(variant: dict[str, object], trades_df: pd.DataFrame) -> dict[str, object]:
    closed = trades_df[trades_df["ExitDate"].notna()].copy()
    closed_trades = int(len(closed))
    win_rate = float((closed["PnLPct"] > 0).mean() * 100) if closed_trades else np.nan
    avg_return = float(closed["PnLPct"].mean()) if closed_trades else np.nan
    median_return = float(closed["PnLPct"].median()) if closed_trades else np.nan
    avg_hold_days = float(closed["HoldDays"].mean()) if closed_trades else np.nan
    annualized = annualize_trade_return(avg_return, avg_hold_days)
    return {
        "Variant": variant["label"],
        "TrendMode": variant["trend_mode"],
        "EntryMode": variant["entry_mode"],
        "MinScore": variant["min_entry_score"],
        "MaxRank": variant["max_entry_rank"],
        "RSvsSPY20dMin": variant["min_rs_vs_spy_20d"],
        "RSvsQQQ20dMin": variant["min_rs_vs_qqq_20d"],
        "ClosedTrades": closed_trades,
        "OpenTrades": int(trades_df["ExitDate"].isna().sum()),
        "WinRatePct": round(win_rate, 2) if pd.notna(win_rate) else np.nan,
        "AvgRetPct": round(avg_return, 2) if pd.notna(avg_return) else np.nan,
        "MedianRetPct": round(median_return, 2) if pd.notna(median_return) else np.nan,
        "AvgHoldDays": round(avg_hold_days, 1) if pd.notna(avg_hold_days) else np.nan,
        "AnnualizedTradePct": round(annualized, 2) if annualized is not None else np.nan,
    }


def run_major_index_backtest_sweep() -> pd.DataFrame:
    universe = system_signal_backtest.load_universe(str(MAJOR_INDEX_ETF_FILE))
    leveraged_flags = system_signal_backtest.build_local_leveraged_flags(universe)
    symbols = universe["Symbol"].tolist()
    close, high, low, volume = system_signal_backtest.download_universe_history(symbols)
    usable_symbols = close.columns.tolist()
    leveraged_flags = leveraged_flags.reindex(usable_symbols).fillna(False)
    macro_state = system_signal_backtest.compute_macro_state(close.index)
    exit_state = system_signal_backtest.compute_exit_state(close, volume)
    state_cache: dict[str, dict[str, pd.DataFrame]] = {}
    rows: list[dict[str, object]] = []

    for variant in MAJOR_INDEX_BACKTEST_VARIANTS:
        trend_mode = str(variant["trend_mode"])
        if trend_mode not in state_cache:
            state_cache[trend_mode] = system_signal_backtest.compute_scanner_state(
                close,
                high,
                low,
                volume,
                leveraged_flags,
                macro_state,
                trend_mode=trend_mode,
            )

        trades_df, _ = system_signal_backtest.simulate_trades(
            state_cache[trend_mode],
            exit_state,
            leveraged_flags,
            macro_state,
            system_signal_backtest.DEFAULT_ANALYSIS_DAYS,
            str(variant["entry_mode"]),
            "exit_only",
            float(variant["min_entry_score"]),
            -1.0,
            int(variant["max_entry_rank"]),
            int(variant["max_new_positions_per_day"]),
            int(variant["exit_confirm_days"]),
            0,
            0.0,
            0.0,
            0.0,
            float(variant["min_rs_vs_spy_20d"]),
            float(variant["min_rs_vs_qqq_20d"]),
            False,
            None,
        )
        rows.append(summarize_backtest_variant(variant, trades_df))

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df
    return results_df.sort_values(["AvgRetPct", "ClosedTrades"], ascending=[False, False]).reset_index(drop=True)


def build_major_index_etf_recommendation(
    filtered_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
) -> tuple[str, str]:
    macro_snapshot = macro_market_status.get_macro_regime_snapshot()
    benchmark_snapshot = macro_market_status.get_benchmark_change_snapshot()

    if not filtered_df.empty:
        top = filtered_df.iloc[0]
        return (
            f"Primary ETF: {top['Symbol']}",
            f"{top['SignalType']} with score {top['Score']:.3f}, RS vs SPY {top['RSvsSPY20d']:.2f}, "
            f"RS vs QQQ {top['RSvsQQQ20d']:.2f}. This is the strongest ETF under the dedicated index-ETF rules.",
        )

    if not ranked_df.empty:
        top = ranked_df.iloc[0]
        return (
            f"Watchlist ETF: {top['Symbol']}",
            f"No ETF passed the active tuned preset, but {top['Symbol']} is the strongest base scanner candidate "
            f"with score {top['Score']:.3f} and signal {top['SignalType']}.",
        )

    regime = str(macro_snapshot.get("regime", "Mixed"))
    spy_20d = float(benchmark_snapshot.get("SPY_20d_pct", float("nan")))
    qqq_20d = float(benchmark_snapshot.get("QQQ_20d_pct", float("nan")))
    if regime == "Risk-on":
        fallback = "QQQ" if pd.notna(qqq_20d) and pd.notna(spy_20d) and qqq_20d > spy_20d else "SPY"
        rationale = "Risk-on regime favors broad equity beta. QQQ wins when Nasdaq 100 is leading; otherwise default to SPY."
    elif regime == "Risk-off":
        fallback = "SGOV"
        rationale = "No clean ETF setup and the macro backdrop is Risk-off, so capital parking beats forcing directional exposure."
    else:
        fallback = "SPY"
        rationale = "Mixed regime and no clean scanner setup favor broad, diversified index exposure over narrower expressions."
    return (f"Fallback Parking ETF: {fallback}", rationale)


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


def render_ranked_scanner(symbols: list[str], trend_mode: str, strategy_preset: str) -> None:
    st.subheader("Ranked Scanner")
    st.write("Rank long candidates across setup, breakout, reclaim, momentum, and leader-continuation states, store a dated snapshot, and show how ranks moved versus the previous run.")
    st.markdown(
        """
        **Principles**

        - The scanner is designed for long ideas with enough liquidity, acceptable extension, and improving or already-strong trend quality.
        - It looks for five states: `SETUP` names near a breakout pivot, `BREAKOUT` names in a fresh controlled breakout, `RECLAIM` names recovering trend quality early, `MOMENTUM` names in a strong breakout burst, and `LEADER` names continuing a dominant uptrend without requiring a brand-new breakout day.
        - Relative strength versus `SPY` and `QQQ` is now a core part of the process, especially for `MOMENTUM` and `LEADER` names.
        - It still blocks names that are too illiquid, too cheap, too hot, too stretched from the 20-day average, too far above the 50-day average when using `ma50` or `hybrid` mode, too far below the 20-day average, structurally weak, or too weak versus the major indexes.
        - Macro market status remains part of the process. `Risk-on` slightly boosts scores, `Risk-off` suppresses them.
        """
    )
    st.markdown(
        """
        **Weighted Score Guide**

        `Score` is the weighted scanner score, or `WScore`. Higher is better, but it is a ranking and readiness tool, not a standalone buy signal.

        - `0.70+`: stronger quality. Trend, relative strength, and location are strongly aligned for that signal type.
        - `0.50` to `0.70`: usable but more mixed. Often a valid candidate, but usually not the cleanest version of the pattern.
        - Below `0.50`: weaker readiness. It may still be improving, but it is lower-priority right now.

        What drives the score:

        - the active signal type: `SETUP`, `BREAKOUT`, `RECLAIM`, `MOMENTUM`, or `LEADER`
        - trend strength from the 50-day versus 150-day averages
        - relative strength versus `SPY` and `QQQ`
        - how stretched price is from the 20-day average, and from the 50-day average in `ma50` and `hybrid` modes
        - proximity to the breakout structure or reclaim trigger
        - breakout freshness or leader persistence, depending on signal type
        - current macro regime, trend, and risk appetite, with a small signal-specific rank adjustment

        `WScore` is now continuous, so it can rise before a hard signal fires and fade before a hard block appears. The actual ranked list still uses hard eligibility rules.

        What is still enforced by hard filters:

        - price and liquidity minimums
        - extension and recent-runup guardrails
        - signal-specific trend and relative-strength requirements
        """
    )
    if strategy_preset == "best_breakout":
        st.info(
            "Best Tuned Breakout preset active: forcing `ma20` trend mode and showing only breakout candidates with "
            "`Score >= 0.60`, `RSvsSPY20d >= 8`, `RSvsQQQ20d >= 2`, capped to the top 10 names."
        )

    if st.button("Run Ranked Scanner"):
        ranked_df, blocked_df = scanner.run_scanner(symbols, trend_mode=trend_mode)
        ranked_df = apply_strategy_preset_to_ranked_df(ranked_df, strategy_preset)
        source_label = "stock_center_best_breakout" if strategy_preset == "best_breakout" else "stock_center"
        ranked_df, blocked_df = scanner.persist_rank_history(ranked_df, blocked_df, source_label=source_label)

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


def render_one_stock_analysis(trend_mode: str, strategy_preset: str) -> None:
    st.subheader("One Stock Analysis")
    st.write("Run the ranked scanner logic on a single ticker, show the current result, and back-calculate its historical WScore over time.")

    symbol = st.text_input("Enter one ticker:", value="AAPL", key="single_stock_symbol").strip().upper()
    major_etf_symbols = get_major_index_etf_symbols()
    is_major_index_etf = symbol in major_etf_symbols
    if strategy_preset == "best_breakout":
        st.info(
            "Best Tuned Breakout preset active: current pass/fail uses breakout-only filtering with `Score >= 0.60`, "
            "`RSvsSPY20d >= 8`, and `RSvsQQQ20d >= 2`. Single-stock analysis cannot enforce universe-wide daily-cap behavior."
        )
    if is_major_index_etf:
        st.info(
            "Major index ETF detected. One Stock Analysis will use the ETF-specific rules here: "
            "`ma20`, `all_candidates`, `Score >= 0.55`, `rank <= 10`, `RSvsSPY20d >= 0`, `RSvsQQQ20d >= -2`."
        )
        st.caption(
            "ETF exit policy reference: "
            f"`{MAJOR_INDEX_ETF_EXIT_POLICY['recommended']}` "
            f"(alternate: `{MAJOR_INDEX_ETF_EXIT_POLICY['alternate']}`)."
        )

    if st.button("Run One Stock Analysis"):
        if not symbol:
            st.error("Enter a ticker first.")
            return

        active_trend_mode = str(MAJOR_INDEX_ETF_DEFAULTS["trend_mode"]) if is_major_index_etf else trend_mode
        ranked_df, blocked_df = scanner.run_scanner([symbol], trend_mode=active_trend_mode)
        if is_major_index_etf:
            filtered_ranked_df = apply_major_index_etf_filter(ranked_df)
        else:
            filtered_ranked_df = apply_strategy_preset_to_ranked_df(ranked_df, strategy_preset)

        if not filtered_ranked_df.empty:
            st.success(f"{symbol} passed the screen.")
            st.dataframe(filtered_ranked_df)
            row = filtered_ranked_df.iloc[0]
            st.metric("WScore", f"{row['Score']:.3f}")
            st.caption(
                f"Signal: {row['SignalType']} | Trade action: {row['TradeAction']} | "
                f"Macro regime: {row['MacroRegime']}"
            )
        else:
            if not ranked_df.empty and is_major_index_etf:
                st.warning(f"{symbol} passed the base scanner but did not pass the major index ETF rules.")
                st.dataframe(ranked_df)
            elif not ranked_df.empty and strategy_preset == "best_breakout":
                st.warning(f"{symbol} passed the base scanner but did not pass the Best Tuned Breakout preset.")
                st.dataframe(ranked_df)
            else:
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
    active_trend_mode = str(MAJOR_INDEX_ETF_DEFAULTS["trend_mode"]) if is_major_index_etf else trend_mode
    state = system_signal_backtest.compute_scanner_state(close, high, low, volume, leveraged_flags, macro_state, trend_mode=active_trend_mode)

    history_df = pd.DataFrame(
        {
            "Date": close.index,
            "Close": close[symbol],
            "WScore": state["raw_score"][symbol],
            "SignalType": state["signal_type"][symbol],
            "Candidate": state["candidate"][symbol],
            "MacroRegime": macro_state["MacroRegime"].values,
            "MacroMarketStatus": (
                macro_state["MacroTrend"].astype(str)
                + " | "
                + macro_state["MacroRisk"].astype(str)
            ).values,
        }
    )
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
    min_date = history_df["Date"].min()
    max_date = history_df["Date"].max()
    shared_x = alt.X(
        "Date:T",
        title=None,
        scale=alt.Scale(domain=[min_date, max_date]),
        axis=alt.Axis(labels=False, ticks=False, domain=False),
    )
    bottom_x = alt.X(
        "Date:T",
        title=None,
        scale=alt.Scale(domain=[min_date, max_date]),
    )

    score_line = (
        alt.Chart(history_df)
        .mark_line(point=True)
        .encode(
            x=shared_x,
            y=alt.Y("WScore:Q", title="WScore"),
            color=alt.value("#1f77b4"),
            tooltip=["Date:T", "Close:Q", "WScore:Q", "SignalType:N", "Candidate:N", "MacroRegime:N", "MacroMarketStatus:N"],
        )
        .properties(height=300)
    )
    candidate_points = (
        alt.Chart(history_df[history_df["Candidate"].fillna(False)])
        .mark_point(size=90, filled=True)
        .encode(
            x=shared_x,
            y=alt.Y("WScore:Q", title="WScore"),
            color=alt.Color("SignalType:N", title="Signal"),
            shape=alt.Shape(
                "SignalType:N",
                title="Signal",
                scale=alt.Scale(
                    domain=["SETUP", "BREAKOUT", "RECLAIM", "MOMENTUM", "LEADER"],
                    range=["diamond", "triangle-up", "square", "circle", "cross"],
                ),
            ),
            tooltip=["Date:T", "Close:Q", "WScore:Q", "SignalType:N", "MacroRegime:N", "MacroMarketStatus:N"],
        )
    )
    score_chart = score_line + candidate_points

    price_chart = (
        alt.Chart(history_df)
        .mark_line()
        .encode(
            x=shared_x,
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
            tooltip=["Date:T", "Close:Q", "WScore:Q", "SignalType:N", "MacroRegime:N", "MacroMarketStatus:N"],
        )
        .properties(height=260)
    )
    price_candidate_points = (
        alt.Chart(history_df[history_df["Candidate"].fillna(False)])
        .mark_point(size=90, filled=True)
        .encode(
            x=shared_x,
            y=alt.Y("Close:Q", title="Close"),
            color=alt.Color("SignalType:N", title="Signal"),
            shape=alt.Shape(
                "SignalType:N",
                title="Signal",
                scale=alt.Scale(
                    domain=["SETUP", "BREAKOUT", "RECLAIM", "MOMENTUM", "LEADER"],
                    range=["diamond", "triangle-up", "square", "circle", "cross"],
                ),
            ),
            tooltip=["Date:T", "Close:Q", "WScore:Q", "SignalType:N", "MacroRegime:N", "MacroMarketStatus:N"],
        )
    )
    price_chart = price_chart + price_candidate_points

    exit_chart = (
        alt.Chart(history_df)
        .mark_line(point=True)
        .encode(
            x=bottom_x,
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
    combined_chart = alt.vconcat(
        score_chart,
        price_chart,
        exit_chart,
        spacing=12,
    ).resolve_scale(x="shared")
    st.altair_chart(combined_chart, use_container_width=True)
    st.dataframe(history_df.sort_values("Date", ascending=False))


def render_position_exit_review(strategy_preset: str) -> None:
    st.subheader("Position Exit Review")
    st.write("Upload your transaction history export and get HOLD / REVIEW / EXIT suggestions for open positions based on the current multi-signal scanner and exit logic.")
    st.markdown(
        """
        **How To Read It**

        - The review now evaluates positions in the context of the updated scanner states: `SETUP`, `BREAKOUT`, `RECLAIM`, `MOMENTUM`, and `LEADER`.
        - `ScannerScore` is now the same continuous readiness-style score used by the updated strategy. It can fade before a hard break and improve before a hard trigger.
        - `HOLD` means the trend and signal context still support the position.
        - `REVIEW` means the position is weakening, getting extended, or losing quality, but is not automatically broken.
        - `EXIT` means the position has tripped stronger failure, loss, or profit-protection conditions.

        What tends to push a position toward `REVIEW` or `EXIT`:

        - price losing the 20-day and 50-day with weakening momentum
        - negative MACD while trading below key moving averages
        - a fresh breakout failing to hold its pivot
        - a hot winner getting too extended and starting to give back strength
        - the scanner score slipping below the hold or review thresholds
        - macro pressure, especially for more fragile setup-style entries
        """
    )
    if strategy_preset == "best_breakout":
        st.info(
            "Best Tuned Breakout preset active: exit review will only treat a position as an active scanner setup when it "
            "still matches the tuned breakout profile (`ma20` structure, breakout hold, score >= 0.60, RS >= 8 vs SPY and >= 2 vs QQQ)."
        )
    txn_file = st.file_uploader("Upload transaction history CSV", type=["csv"], key="txn_history")

    if st.button("Run Position Exit Review"):
        if txn_file is None:
            st.error("Upload a transaction history CSV first.")
            return

        with NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(txn_file.getvalue())
            tmp_path = tmp_file.name

        review_df = position_exit_review.review_positions_from_csv(tmp_path, strategy_preset=strategy_preset)
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


def render_major_index_etf_lab(trend_mode: str, strategy_preset: str) -> None:
    st.subheader("Major Index ETF")
    st.write(
        "Scan a curated major-index ETF universe, generate a fallback parking recommendation when single-stock "
        "relative strength is weak, and compare a compact backtest sweep for ETF-specific variants."
    )
    st.info(
        "This panel uses ETF-specific defaults instead of the stock preset: "
        "`ma20`, `all_candidates`, `Score >= 0.55`, `rank <= 10`, `RSvsSPY20d >= 0`, `RSvsQQQ20d >= -2`."
    )
    st.markdown(
        f"""
        **Recommended ETF Exit Policy**

        - Primary policy: `{MAJOR_INDEX_ETF_EXIT_POLICY["recommended"]}`
        - Alternate policy: `{MAJOR_INDEX_ETF_EXIT_POLICY["alternate"]}`

        Why:

        - major index ETFs backtested better with slower exit confirmation than single-stock review logic
        - `review_or_exit` was too sensitive and cut returns sharply
        - the ETF study favored waiting for full `EXIT` conditions instead of reacting to early `REVIEW` states
        """
    )

    try:
        universe = load_major_index_etf_universe()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.caption(f"Universe file: {MAJOR_INDEX_ETF_FILE.name} | Symbols loaded: {len(universe)}")
    with st.expander("Major Index ETF Universe"):
        st.dataframe(universe)

    if st.button("Run Major Index ETF Scan"):
        with st.spinner("Scanning major index ETFs..."):
            ranked_df, blocked_df = scanner.run_scanner(
                universe["Symbol"].tolist(),
                trend_mode=str(MAJOR_INDEX_ETF_DEFAULTS["trend_mode"]),
            )
            ranked_df = scanner.add_rank_columns(ranked_df)
            active_df = ranked_df[
                (ranked_df["Score"] >= float(MAJOR_INDEX_ETF_DEFAULTS["min_entry_score"]))
                & (ranked_df["Rank"] <= int(MAJOR_INDEX_ETF_DEFAULTS["max_entry_rank"]))
                & (ranked_df["RSvsSPY20d"] >= float(MAJOR_INDEX_ETF_DEFAULTS["min_rs_vs_spy_20d"]))
                & (ranked_df["RSvsQQQ20d"] >= float(MAJOR_INDEX_ETF_DEFAULTS["min_rs_vs_qqq_20d"]))
            ].copy()
            active_df = active_df.sort_values(["Score", "RSvsSPY20d", "RSvsQQQ20d"], ascending=[False, False, False])
            title, message = build_major_index_etf_recommendation(active_df, ranked_df)

        st.write(f"Base scanner candidates: {len(ranked_df)}")
        st.write(f"ETF-rule qualified candidates: {len(active_df)}")
        st.info(f"{title}: {message}")

        st.write("Active recommendation set")
        st.dataframe(active_df if not active_df.empty else ranked_df)

        st.write("Blocked symbols")
        st.dataframe(blocked_df)

    if st.button("Run Major Index ETF Backtest Sweep"):
        with st.spinner("Running ETF backtest sweep..."):
            results_df = run_major_index_backtest_sweep()

        if results_df.empty:
            st.warning("No backtest results were produced.")
            return

        eligible = results_df[results_df["ClosedTrades"] >= 5].copy()
        best_row = (
            eligible.sort_values(["AvgRetPct", "ClosedTrades"], ascending=[False, False]).iloc[0]
            if not eligible.empty
            else results_df.iloc[0]
        )
        st.success(
            f"Best practical variant: {best_row['Variant']} | avg return {best_row['AvgRetPct']:.2f}% | "
            f"closed trades {int(best_row['ClosedTrades'])} | annualized trade proxy {best_row['AnnualizedTradePct']:.2f}%"
        )
        st.caption(
            "Use the ETF-specific exit policy with that entry variant: "
            f"`{MAJOR_INDEX_ETF_EXIT_POLICY['recommended']}`. "
            f"Alternate: `{MAJOR_INDEX_ETF_EXIT_POLICY['alternate']}`."
        )
        st.dataframe(results_df)


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
            "Major Index ETF",
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
    trend_mode = st.sidebar.selectbox(
        "Trend mode",
        options=list(scanner.TREND_MODES),
        index=list(scanner.TREND_MODES).index("ma20"),
        help="ma50 = current baseline, ma20 = more sensitive, hybrid = MA50 structure plus MA20 timing.",
    )
    strategy_preset = st.sidebar.selectbox(
        "Strategy preset",
        options=list(STRATEGY_PRESETS.keys()),
        format_func=lambda key: STRATEGY_PRESETS[key]["label"],
        index=list(STRATEGY_PRESETS.keys()).index("classic"),
        help="Use the full scanner for market participation. Select breakout-only only when you explicitly want very sparse, high-relative-strength entries.",
    )
    st.sidebar.caption(STRATEGY_PRESETS[strategy_preset]["description"])
    effective_trend_mode = resolve_trend_mode(trend_mode, strategy_preset)
    if effective_trend_mode != trend_mode:
        st.sidebar.caption(f"Active trend mode overridden to `{effective_trend_mode}` by the selected preset.")
    csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
    symbols = load_symbols_from_inputs(tickers_input, csv_file)

    st.caption(f"Symbols loaded: {len(symbols)}")

    if analysis_choice == "Ranked Scanner":
        render_ranked_scanner(symbols, effective_trend_mode, strategy_preset)
    elif analysis_choice == "Major Index ETF":
        render_major_index_etf_lab(effective_trend_mode, strategy_preset)
    elif analysis_choice == "One Stock Analysis":
        render_one_stock_analysis(effective_trend_mode, strategy_preset)
    elif analysis_choice == "Ranking History":
        render_ranking_history()
    elif analysis_choice == "Macro Market Status":
        render_macro_market_status()
    elif analysis_choice == "MACD Screening":
        render_macd_screening(symbols)
    elif analysis_choice == "ATR Finder":
        render_atr_finder(symbols)
    elif analysis_choice == "Position Exit Review":
        render_position_exit_review(strategy_preset)
    else:
        render_news_sentiment(symbols)


if __name__ == "__main__":
    main()
