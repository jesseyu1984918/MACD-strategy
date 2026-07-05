from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import system_signal_backtest


DEFAULT_ANALYSIS_DAYS = 252
DEFAULT_SHORT_WINDOW = 6
DEFAULT_LONG_WINDOW = 19
DEFAULT_SIGNAL_WINDOW = 9
DEFAULT_TRADES_OUT = "macd_zero_cross_trades.csv"
DEFAULT_DAILY_OUT = "macd_zero_cross_daily.csv"
DEFAULT_SUMMARY_OUT = "macd_zero_cross_summary.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the MACD crossover rule: buy below zero, sell above zero."
    )
    parser.add_argument("--universe", default="dataset/combined_universe.csv")
    parser.add_argument("--analysis-days", type=int, default=DEFAULT_ANALYSIS_DAYS)
    parser.add_argument("--short-window", type=int, default=DEFAULT_SHORT_WINDOW)
    parser.add_argument("--long-window", type=int, default=DEFAULT_LONG_WINDOW)
    parser.add_argument("--signal-window", type=int, default=DEFAULT_SIGNAL_WINDOW)
    parser.add_argument("--trades-out", default=DEFAULT_TRADES_OUT)
    parser.add_argument("--daily-out", default=DEFAULT_DAILY_OUT)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    return parser.parse_args()


def compute_macd_state(
    close: pd.DataFrame,
    short_window: int,
    long_window: int,
    signal_window: int,
) -> dict[str, pd.DataFrame]:
    ema_short = close.ewm(span=short_window, adjust=False).mean()
    ema_long = close.ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    entry_signal = (
        (macd > signal)
        & (macd.shift(1) <= signal.shift(1))
        & (macd < 0)
        & (signal < 0)
    )
    exit_signal = (
        (macd < signal)
        & (macd.shift(1) >= signal.shift(1))
        & (macd > 0)
        & (signal > 0)
    )

    return {
        "close": close,
        "macd": macd,
        "signal": signal,
        "entry_signal": entry_signal.fillna(False),
        "exit_signal": exit_signal.fillna(False),
    }


def simulate_macd_strategy(
    state: dict[str, pd.DataFrame],
    analysis_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = state["close"]
    entry_signal = state["entry_signal"]
    exit_signal = state["exit_signal"]
    macd = state["macd"]
    signal = state["signal"]

    analysis_dates = close.index[-analysis_days:] if len(close.index) > analysis_days else close.index
    if analysis_dates.empty:
        return pd.DataFrame(), pd.DataFrame()

    prev_close = close.shift(1)
    open_positions: dict[str, dict[str, object]] = {}
    trades: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    equity = 1.0

    for current_date in analysis_dates:
        positions_at_open = list(open_positions.keys())
        position_count = len(positions_at_open)

        if position_count > 0:
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

        exit_symbols: list[str] = []
        for symbol, position in open_positions.items():
            if not bool(exit_signal.at[current_date, symbol]):
                continue

            exit_price = close.at[current_date, symbol]
            if pd.isna(exit_price):
                continue

            hold_days = int((current_date - position["entry_date"]).days)
            pnl_pct = ((float(exit_price) / float(position["entry_price"])) - 1.0) * 100
            trades.append(
                {
                    "Symbol": symbol,
                    "EntryDate": position["entry_date"].date(),
                    "ExitDate": current_date.date(),
                    "EntryPrice": round(float(position["entry_price"]), 4),
                    "ExitPrice": round(float(exit_price), 4),
                    "PnLPct": round(float(pnl_pct), 2),
                    "HoldDays": hold_days,
                    "EntryMACD": round(float(position["entry_macd"]), 4),
                    "EntrySignal": round(float(position["entry_signal"]), 4),
                    "ExitMACD": round(float(macd.at[current_date, symbol]), 4),
                    "ExitSignal": round(float(signal.at[current_date, symbol]), 4),
                    "ExitReason": "macd_down_cross_above_zero",
                }
            )
            exit_symbols.append(symbol)

        for symbol in exit_symbols:
            open_positions.pop(symbol, None)

        entry_symbols = entry_signal.loc[current_date]
        entries_opened = 0
        for symbol in entry_symbols[entry_symbols].index.tolist():
            if symbol in open_positions:
                continue

            entry_price = close.at[current_date, symbol]
            if pd.isna(entry_price):
                continue

            open_positions[symbol] = {
                "entry_date": current_date,
                "entry_price": float(entry_price),
                "entry_macd": float(macd.at[current_date, symbol]),
                "entry_signal": float(signal.at[current_date, symbol]),
            }
            entries_opened += 1

        daily_rows.append(
            {
                "Date": current_date.date(),
                "OpenPositions": len(positions_at_open),
                "Entries": entries_opened,
                "Exits": len(exit_symbols),
                "PortfolioDayReturnPct": round(portfolio_day_return * 100, 4),
                "Equity": round(equity, 6),
            }
        )

    final_date = analysis_dates[-1]
    for symbol, position in open_positions.items():
        last_price = close.at[final_date, symbol]
        if pd.isna(last_price):
            continue
        hold_days = int((final_date - position["entry_date"]).days)
        pnl_pct = ((float(last_price) / float(position["entry_price"])) - 1.0) * 100
        trades.append(
            {
                "Symbol": symbol,
                "EntryDate": position["entry_date"].date(),
                "ExitDate": None,
                "EntryPrice": round(float(position["entry_price"]), 4),
                "ExitPrice": round(float(last_price), 4),
                "PnLPct": round(float(pnl_pct), 2),
                "HoldDays": hold_days,
                "EntryMACD": round(float(position["entry_macd"]), 4),
                "EntrySignal": round(float(position["entry_signal"]), 4),
                "ExitMACD": round(float(macd.at[final_date, symbol]), 4),
                "ExitSignal": round(float(signal.at[final_date, symbol]), 4),
                "ExitReason": "open_at_end",
            }
        )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["EntryDate", "Symbol"]).reset_index(drop=True)

    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_df = daily_df.sort_values("Date").reset_index(drop=True)

    return trades_df, daily_df


def build_summary(
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    universe_size: int,
    valid_symbols: int,
    analysis_days: int,
    short_window: int,
    long_window: int,
    signal_window: int,
) -> str:
    closed = trades_df[trades_df["ExitReason"].ne("open_at_end")].copy() if not trades_df.empty else pd.DataFrame()
    open_trades = trades_df[trades_df["ExitReason"].eq("open_at_end")].copy() if not trades_df.empty else pd.DataFrame()

    lines = [
        "MACD Zero-Cross Backtest",
        f"Universe requested: {universe_size}",
        f"Universe with usable history: {valid_symbols}",
        f"Analysis window (trading days): {analysis_days}",
        f"MACD windows: fast={short_window}, slow={long_window}, signal={signal_window}",
        "Entry rule: MACD crosses above signal while both are below zero",
        "Exit rule: MACD crosses below signal while both are above zero",
        f"Trades generated: {len(trades_df)}",
        f"Closed trades: {len(closed)}",
        f"Open trades at end: {len(open_trades)}",
    ]

    if not daily_df.empty:
        equity = daily_df["Equity"].astype(float)
        total_return_pct = (equity.iloc[-1] - 1.0) * 100
        rolling_peak = equity.cummax()
        max_drawdown_pct = ((equity / rolling_peak) - 1.0).min() * 100
        day_returns = daily_df["PortfolioDayReturnPct"].astype(float) / 100.0
        sharpe = 0.0
        if day_returns.std(ddof=0) > 0:
            sharpe = float((day_returns.mean() / day_returns.std(ddof=0)) * np.sqrt(252))
        lines.extend(
            [
                f"Portfolio total return: {total_return_pct:.2f}%",
                f"Portfolio max drawdown: {max_drawdown_pct:.2f}%",
                f"Portfolio daily Sharpe (rf=0): {sharpe:.2f}",
            ]
        )

    if closed.empty:
        lines.append("No closed trades were generated.")
        return "\n".join(lines)

    win_rate = (closed["PnLPct"] > 0).mean() * 100
    avg_return = closed["PnLPct"].mean()
    median_return = closed["PnLPct"].median()
    avg_hold = closed["HoldDays"].mean()

    lines.extend(
        [
            f"Win rate: {win_rate:.2f}%",
            f"Average return per closed trade: {avg_return:.2f}%",
            f"Median return per closed trade: {median_return:.2f}%",
            f"Average hold days: {avg_hold:.1f}",
            "",
            "Top 10 closed trades by return:",
            closed.sort_values("PnLPct", ascending=False)
            .head(10)[["Symbol", "EntryDate", "ExitDate", "PnLPct", "HoldDays"]]
            .to_string(index=False),
            "",
            "Bottom 10 closed trades by return:",
            closed.sort_values("PnLPct", ascending=True)
            .head(10)[["Symbol", "EntryDate", "ExitDate", "PnLPct", "HoldDays"]]
            .to_string(index=False),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    system_signal_backtest.YF_CACHE_DIR.mkdir(exist_ok=True)
    system_signal_backtest.yf_cache.set_cache_location(str(system_signal_backtest.YF_CACHE_DIR))

    universe = system_signal_backtest.load_universe(args.universe)
    symbols = universe["Symbol"].tolist()
    close, _high, _low, _volume = system_signal_backtest.download_universe_history(symbols)

    state = compute_macd_state(
        close,
        short_window=args.short_window,
        long_window=args.long_window,
        signal_window=args.signal_window,
    )
    trades_df, daily_df = simulate_macd_strategy(state, args.analysis_days)

    trades_df.to_csv(args.trades_out, index=False)
    daily_df.to_csv(args.daily_out, index=False)

    summary = build_summary(
        trades_df,
        daily_df,
        universe_size=len(symbols),
        valid_symbols=len(close.columns),
        analysis_days=args.analysis_days,
        short_window=args.short_window,
        long_window=args.long_window,
        signal_window=args.signal_window,
    )
    Path(args.summary_out).write_text(summary)

    print(summary)
    print(f"\nSaved trades to {args.trades_out}")
    print(f"Saved daily results to {args.daily_out}")
    print(f"Saved summary to {args.summary_out}")


if __name__ == "__main__":
    main()
