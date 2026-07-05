from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import system_signal_backtest


DEFAULT_TRADES_OUT = "scanner_portfolio_trades.csv"
DEFAULT_SIGNALS_OUT = "scanner_portfolio_signals.csv"
DEFAULT_DAILY_OUT = "scanner_portfolio_daily.csv"
DEFAULT_SUMMARY_OUT = "scanner_portfolio_summary.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scanner backtest and compute a portfolio equity curve from executed positions."
    )
    parser.add_argument("--universe", default="dataset/combined_universe.csv")
    parser.add_argument("--analysis-days", type=int, default=system_signal_backtest.DEFAULT_ANALYSIS_DAYS)
    parser.add_argument("--trend-mode", choices=list(system_signal_backtest.scanner.TREND_MODES), default="ma50")
    parser.add_argument("--allowed-entry-signals", default="ALL")
    parser.add_argument("--entry-mode", choices=["all_candidates", "breakout_only"], default="breakout_only")
    parser.add_argument("--exit-on", choices=["review_or_exit", "exit_only"], default="review_or_exit")
    parser.add_argument("--min-entry-score", type=float, default=0.0)
    parser.add_argument("--leader-min-entry-score", type=float, default=-1.0)
    parser.add_argument("--max-entry-rank", type=int, default=0)
    parser.add_argument("--max-new-positions-per-day", type=int, default=0)
    parser.add_argument("--exit-confirm-days", type=int, default=1)
    parser.add_argument("--early-failure-days", type=int, default=0)
    parser.add_argument("--early-failure-loss-pct", type=float, default=0.0)
    parser.add_argument("--profit-lock-trigger-pct", type=float, default=0.0)
    parser.add_argument("--profit-lock-drawdown-pct", type=float, default=0.0)
    parser.add_argument("--min-rs-vs-spy-20d", type=float, default=-999.0)
    parser.add_argument("--min-rs-vs-qqq-20d", type=float, default=-999.0)
    parser.add_argument("--allow-risk-off-entries", action="store_true")
    parser.add_argument("--trades-out", default=DEFAULT_TRADES_OUT)
    parser.add_argument("--signals-out", default=DEFAULT_SIGNALS_OUT)
    parser.add_argument("--daily-out", default=DEFAULT_DAILY_OUT)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    return parser.parse_args()


def simulate_trades_with_equity(
    state: dict[str, pd.DataFrame],
    exit_state: dict[str, pd.DataFrame],
    leveraged_flags: pd.Series,
    macro_state: pd.DataFrame,
    analysis_days: int,
    entry_mode: str,
    exit_on: str,
    min_entry_score: float,
    leader_min_entry_score: float,
    max_entry_rank: int,
    max_new_positions_per_day: int,
    exit_confirm_days: int,
    early_failure_days: int,
    early_failure_loss_pct: float,
    profit_lock_trigger_pct: float,
    profit_lock_drawdown_pct: float,
    min_rs_vs_spy_20d: float,
    min_rs_vs_qqq_20d: float,
    allow_risk_off_entries: bool,
    allowed_entry_signals: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate = state["candidate"]
    signal_type_state = state["signal_type"]
    score = state["score"]
    daily_rank = state["daily_rank"]
    close = state["close"]
    prev_close = close.shift(1)

    dates = candidate.index
    analysis_dates = dates[-analysis_days:] if len(dates) > analysis_days else dates

    open_positions: dict[str, dict[str, object]] = {}
    trades: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    equity = 1.0

    for current_date in analysis_dates:
        positions_at_open = list(open_positions.keys())
        if positions_at_open:
            day_returns: list[float] = []
            for symbol in positions_at_open:
                last_close = prev_close.at[current_date, symbol]
                current_close = close.at[current_date, symbol]
                if pd.isna(last_close) or pd.isna(current_close) or float(last_close) == 0.0:
                    continue
                day_returns.append((float(current_close) / float(last_close)) - 1.0)
            portfolio_day_return = float(np.mean(day_returns)) if day_returns else 0.0
        else:
            portfolio_day_return = 0.0
        equity *= 1.0 + portfolio_day_return

        todays_entries = score.loc[current_date].dropna().sort_values(ascending=False).index.tolist()

        entries_opened_today = 0
        for symbol in todays_entries:
            if symbol in open_positions:
                continue

            entry_price = close.at[current_date, symbol]
            if pd.isna(entry_price):
                continue

            signal_type = str(signal_type_state.at[current_date, symbol])
            if allowed_entry_signals and signal_type not in allowed_entry_signals:
                continue
            trade_action = system_signal_backtest.scanner.build_trade_action(
                signal_type, 0.0 if signal_type == "BREAKOUT" else np.nan, 0.0
            )
            if entry_mode == "breakout_only" and trade_action != "BUY":
                continue

            entry_rank = daily_rank.at[current_date, symbol]
            entry_score = score.at[current_date, symbol]
            macro_regime = macro_state.at[current_date, "MacroRegime"]
            if pd.isna(entry_score):
                continue
            required_entry_score = (
                leader_min_entry_score
                if signal_type == "LEADER" and leader_min_entry_score >= 0
                else min_entry_score
            )
            if entry_score < required_entry_score:
                continue
            if max_entry_rank > 0 and pd.notna(entry_rank) and int(entry_rank) > max_entry_rank:
                continue
            if not allow_risk_off_entries and macro_regime == "Risk-off":
                continue
            stock_20d = state["stock_change_20d_pct"].at[current_date, symbol]
            spy_20d = macro_state.at[current_date, "SPY20dPct"]
            qqq_20d = macro_state.at[current_date, "QQQ20dPct"]
            rs_vs_spy_20d = stock_20d - spy_20d if pd.notna(stock_20d) and pd.notna(spy_20d) else np.nan
            rs_vs_qqq_20d = stock_20d - qqq_20d if pd.notna(stock_20d) and pd.notna(qqq_20d) else np.nan
            if pd.notna(rs_vs_spy_20d) and rs_vs_spy_20d < min_rs_vs_spy_20d:
                continue
            if pd.notna(rs_vs_qqq_20d) and rs_vs_qqq_20d < min_rs_vs_qqq_20d:
                continue
            if max_new_positions_per_day > 0 and entries_opened_today >= max_new_positions_per_day:
                break

            open_positions[symbol] = {
                "entry_date": current_date,
                "entry_price": float(entry_price),
                "entry_signal_type": signal_type,
                "entry_rank": int(entry_rank) if pd.notna(entry_rank) else np.nan,
                "entry_score": round(float(entry_score), 3),
                "pending_exit_days": 0,
                "peak_pnl_pct": 0.0,
            }
            entries_opened_today += 1
            signal_rows.append(
                {
                    "Date": current_date.date(),
                    "Symbol": symbol,
                    "Action": "BUY",
                    "SignalType": signal_type,
                    "Rank": int(entry_rank) if pd.notna(entry_rank) else np.nan,
                    "Score": round(float(entry_score), 3),
                    "Price": round(float(entry_price), 4),
                    "Recommendation": None,
                    "Reason": "scanner_candidate",
                    "MacroRegime": macro_regime,
                    "RSvsSPY20d": round(float(rs_vs_spy_20d), 2) if pd.notna(rs_vs_spy_20d) else np.nan,
                    "RSvsQQQ20d": round(float(rs_vs_qqq_20d), 2) if pd.notna(rs_vs_qqq_20d) else np.nan,
                }
            )

        exit_symbols: list[str] = []
        for symbol, position in open_positions.items():
            if current_date <= position["entry_date"]:
                continue

            last_price = exit_state["close"].at[current_date, symbol]
            if pd.isna(last_price):
                continue
            pnl_pct = ((float(last_price) / float(position["entry_price"])) - 1) * 100
            hold_days = int((current_date - position["entry_date"]).days)
            position["peak_pnl_pct"] = max(float(position.get("peak_pnl_pct", pnl_pct)), float(pnl_pct))

            review_row = pd.Series(
                {
                    "Side": "LONG",
                    "LastPrice": float(last_price),
                    "MA20": exit_state["ma20"].at[current_date, symbol],
                    "MA50": exit_state["ma50"].at[current_date, symbol],
                    "MA150": exit_state["ma150"].at[current_date, symbol],
                    "PnLPct": pnl_pct,
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
                    "ScannerScore": state["score"].at[current_date, symbol],
                    "ScannerTrend": state["trend"].at[current_date, symbol],
                    "ScannerSignalType": str(state["signal_type"].at[current_date, symbol]),
                }
            )
            macro_snapshot = {
                "regime": macro_state.at[current_date, "MacroRegime"],
                "score": macro_state.at[current_date, "MacroScore"],
                "scanner_multiplier": macro_state.at[current_date, "ScannerMultiplier"],
                "exit_review_threshold_shift": macro_state.at[current_date, "ExitReviewThresholdShift"],
            }

            recommendation, reason = system_signal_backtest.build_recommendation(
                review_row,
                bool(leveraged_flags.get(symbol, False)),
                macro_snapshot,
            )
            if early_failure_days > 0 and hold_days <= early_failure_days and pnl_pct <= -abs(early_failure_loss_pct):
                recommendation = "EXIT"
                reason = f"early_failure_stop_{early_failure_days}d_{abs(early_failure_loss_pct):.1f}pct; {reason}"
            peak_pnl_pct = float(position.get("peak_pnl_pct", pnl_pct))
            if (
                profit_lock_trigger_pct > 0
                and profit_lock_drawdown_pct > 0
                and peak_pnl_pct >= profit_lock_trigger_pct
                and pnl_pct <= peak_pnl_pct - profit_lock_drawdown_pct
            ):
                recommendation = "EXIT"
                reason = f"profit_lock_{profit_lock_trigger_pct:.1f}_{profit_lock_drawdown_pct:.1f}; {reason}"

            exit_signal = recommendation != "HOLD" if exit_on == "review_or_exit" else recommendation == "EXIT"
            if exit_signal:
                position["pending_exit_days"] = int(position.get("pending_exit_days", 0)) + 1
            else:
                position["pending_exit_days"] = 0
            should_exit = bool(position["pending_exit_days"] >= max(exit_confirm_days, 1))
            if not should_exit:
                continue

            exit_symbols.append(symbol)
            trades.append(
                {
                    "Symbol": symbol,
                    "EntryDate": position["entry_date"].date(),
                    "ExitDate": current_date.date(),
                    "EntrySignalType": position["entry_signal_type"],
                    "EntryRank": position["entry_rank"],
                    "EntryScore": position["entry_score"],
                    "EntryPrice": round(float(position["entry_price"]), 4),
                    "ExitPrice": round(float(last_price), 4),
                    "PnLPct": round(float(pnl_pct), 2),
                    "HoldDays": hold_days,
                    "ExitRecommendation": recommendation,
                    "ExitReason": reason,
                }
            )
            signal_rows.append(
                {
                    "Date": current_date.date(),
                    "Symbol": symbol,
                    "Action": "SELL",
                    "SignalType": position["entry_signal_type"],
                    "Rank": position["entry_rank"],
                    "Score": position["entry_score"],
                    "Price": round(float(last_price), 4),
                    "Recommendation": recommendation,
                    "Reason": reason,
                    "MacroRegime": macro_snapshot["regime"],
                }
            )

        for symbol in exit_symbols:
            open_positions.pop(symbol, None)

        daily_rows.append(
            {
                "Date": current_date.date(),
                "OpenPositions": len(positions_at_open),
                "Entries": entries_opened_today,
                "Exits": len(exit_symbols),
                "PortfolioDayReturnPct": round(portfolio_day_return * 100, 4),
                "Equity": round(equity, 6),
            }
        )

    if analysis_dates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    final_date = analysis_dates[-1]
    for symbol, position in open_positions.items():
        last_price = close.at[final_date, symbol]
        if pd.isna(last_price):
            continue
        hold_days = int((final_date - position["entry_date"]).days)
        pnl_pct = ((float(last_price) / float(position["entry_price"])) - 1) * 100
        trades.append(
            {
                "Symbol": symbol,
                "EntryDate": position["entry_date"].date(),
                "ExitDate": None,
                "EntrySignalType": position["entry_signal_type"],
                "EntryRank": position["entry_rank"],
                "EntryScore": position["entry_score"],
                "EntryPrice": round(float(position["entry_price"]), 4),
                "ExitPrice": round(float(last_price), 4),
                "PnLPct": round(float(pnl_pct), 2),
                "HoldDays": hold_days,
                "ExitRecommendation": "OPEN",
                "ExitReason": "still_open_at_end_of_backtest",
            }
        )

    trades_df = pd.DataFrame(trades).sort_values(["EntryDate", "Symbol"]).reset_index(drop=True) if trades else pd.DataFrame()
    signals_df = (
        pd.DataFrame(signal_rows).sort_values(["Date", "Action", "Symbol"]).reset_index(drop=True)
        if signal_rows
        else pd.DataFrame()
    )
    daily_df = pd.DataFrame(daily_rows).sort_values("Date").reset_index(drop=True) if daily_rows else pd.DataFrame()
    return trades_df, signals_df, daily_df


def build_portfolio_summary(
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    universe_size: int,
    valid_symbols: int,
    args: argparse.Namespace,
) -> str:
    scanner_summary = system_signal_backtest.build_summary(
        trades_df,
        universe_size,
        valid_symbols,
        args.analysis_days,
        args.trend_mode,
        args.entry_mode,
        args.exit_on,
        args.min_entry_score,
        args.leader_min_entry_score,
        args.max_entry_rank,
        args.max_new_positions_per_day,
        args.exit_confirm_days,
        args.early_failure_days,
        args.early_failure_loss_pct,
        args.profit_lock_trigger_pct,
        args.profit_lock_drawdown_pct,
        args.min_rs_vs_spy_20d,
        args.min_rs_vs_qqq_20d,
        args.allow_risk_off_entries,
    )

    portfolio_lines = ["", "Portfolio overlay:"]
    if daily_df.empty:
        portfolio_lines.append("No daily portfolio data available.")
        return scanner_summary + "\n" + "\n".join(portfolio_lines)

    equity = daily_df["Equity"].astype(float)
    open_positions = daily_df["OpenPositions"].astype(float)
    total_return_pct = (equity.iloc[-1] - 1.0) * 100
    rolling_peak = equity.cummax()
    max_drawdown_pct = ((equity / rolling_peak) - 1.0).min() * 100
    day_returns = daily_df["PortfolioDayReturnPct"].astype(float) / 100.0
    sharpe = 0.0
    if day_returns.std(ddof=0) > 0:
        sharpe = float((day_returns.mean() / day_returns.std(ddof=0)) * np.sqrt(252))

    portfolio_lines.extend(
        [
            "Equal-weight across positions already open at each session.",
            f"Portfolio total return: {total_return_pct:.2f}%",
            f"Portfolio max drawdown: {max_drawdown_pct:.2f}%",
            f"Portfolio daily Sharpe (rf=0): {sharpe:.2f}",
            f"Average open positions: {open_positions.mean():.1f}",
            f"Max open positions: {int(open_positions.max()) if not open_positions.empty else 0}",
        ]
    )
    return scanner_summary + "\n" + "\n".join(portfolio_lines)


def main() -> None:
    args = parse_args()
    system_signal_backtest.YF_CACHE_DIR.mkdir(exist_ok=True)
    system_signal_backtest.yf_cache.set_cache_location(str(system_signal_backtest.YF_CACHE_DIR))

    universe = system_signal_backtest.load_universe(args.universe)
    leveraged_flags = system_signal_backtest.build_local_leveraged_flags(universe)
    symbols = universe["Symbol"].tolist()

    close, high, low, volume = system_signal_backtest.download_universe_history(symbols)
    usable_symbols = close.columns.tolist()
    leveraged_flags = leveraged_flags.reindex(usable_symbols).fillna(False)

    macro_state = system_signal_backtest.compute_macro_state(close.index)
    state = system_signal_backtest.compute_scanner_state(
        close, high, low, volume, leveraged_flags, macro_state, trend_mode=args.trend_mode
    )
    exit_state = system_signal_backtest.compute_exit_state(close, volume)

    allowed_entry_signals = None
    if str(args.allowed_entry_signals).strip().upper() != "ALL":
        allowed_entry_signals = {
            token.strip().upper()
            for token in str(args.allowed_entry_signals).split(",")
            if token.strip()
        }

    trades_df, signals_df, daily_df = simulate_trades_with_equity(
        state,
        exit_state,
        leveraged_flags,
        macro_state,
        args.analysis_days,
        args.entry_mode,
        args.exit_on,
        args.min_entry_score,
        args.leader_min_entry_score,
        args.max_entry_rank,
        args.max_new_positions_per_day,
        args.exit_confirm_days,
        args.early_failure_days,
        args.early_failure_loss_pct,
        args.profit_lock_trigger_pct,
        args.profit_lock_drawdown_pct,
        args.min_rs_vs_spy_20d,
        args.min_rs_vs_qqq_20d,
        args.allow_risk_off_entries,
        allowed_entry_signals,
    )

    trades_df.to_csv(args.trades_out, index=False)
    signals_df.to_csv(args.signals_out, index=False)
    daily_df.to_csv(args.daily_out, index=False)

    summary = build_portfolio_summary(trades_df, daily_df, len(symbols), len(usable_symbols), args)
    Path(args.summary_out).write_text(summary)

    print(summary)
    print(f"\nSaved trades to {args.trades_out}")
    print(f"Saved signals to {args.signals_out}")
    print(f"Saved daily results to {args.daily_out}")
    print(f"Saved summary to {args.summary_out}")


if __name__ == "__main__":
    main()
