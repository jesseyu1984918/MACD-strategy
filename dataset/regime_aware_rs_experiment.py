from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = REPO_ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

import scanner
import system_signal_backtest
from position_exit_review import build_recommendation


def summarize(name: str, trades_df: pd.DataFrame) -> dict[str, object]:
    closed = trades_df[trades_df["ExitDate"].notna()].copy() if not trades_df.empty else trades_df
    closed_trades = len(closed)
    win_rate = float((closed["PnLPct"] > 0).mean() * 100) if closed_trades else np.nan
    avg_ret = float(closed["PnLPct"].mean()) if closed_trades else np.nan
    median_ret = float(closed["PnLPct"].median()) if closed_trades else np.nan
    avg_hold = float(closed["HoldDays"].mean()) if closed_trades else np.nan
    annualized = (
        (((1 + (avg_ret / 100.0)) ** (252.0 / avg_hold)) - 1) * 100.0
        if closed_trades and avg_hold > 0 and avg_ret > -100
        else np.nan
    )
    return {
        "variant": name,
        "closed_trades": closed_trades,
        "open_trades": int(trades_df["ExitDate"].isna().sum()) if not trades_df.empty else 0,
        "win_rate_pct": round(win_rate, 2) if pd.notna(win_rate) else np.nan,
        "avg_ret_pct": round(avg_ret, 2) if pd.notna(avg_ret) else np.nan,
        "median_ret_pct": round(median_ret, 2) if pd.notna(median_ret) else np.nan,
        "avg_hold_days": round(avg_hold, 1) if pd.notna(avg_hold) else np.nan,
        "annualized_trade_pct": round(annualized, 2) if pd.notna(annualized) else np.nan,
    }


def simulate_regime_aware(
    state: dict[str, pd.DataFrame],
    exit_state: dict[str, pd.DataFrame],
    leveraged_flags: pd.Series,
    macro_state: pd.DataFrame,
    score_min: float,
    max_rank: int,
    max_new_positions_per_day: int,
    exit_confirm_days: int,
    threshold_map: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    candidate = state["candidate"]
    signal_type_state = state["signal_type"]
    score = state["score"]
    daily_rank = state["daily_rank"]
    close = state["close"]
    stock_change_20d_pct = state["stock_change_20d_pct"]

    dates = candidate.index
    analysis_dates = dates[-system_signal_backtest.DEFAULT_ANALYSIS_DAYS:] if len(dates) > system_signal_backtest.DEFAULT_ANALYSIS_DAYS else dates

    open_positions: dict[str, dict[str, object]] = {}
    trades: list[dict[str, object]] = []

    for current_date in analysis_dates:
        todays_entries = score.loc[current_date].dropna().sort_values(ascending=False).index.tolist()
        entries_opened_today = 0

        for symbol in todays_entries:
            if symbol in open_positions:
                continue

            entry_price = close.at[current_date, symbol]
            if pd.isna(entry_price):
                continue

            signal_type = str(signal_type_state.at[current_date, symbol])
            trade_action = scanner.build_trade_action(signal_type, 0.0 if signal_type == "BREAKOUT" else np.nan, 0.0)
            if trade_action != "BUY":
                continue

            entry_rank = daily_rank.at[current_date, symbol]
            entry_score = score.at[current_date, symbol]
            if pd.isna(entry_score) or entry_score < score_min:
                continue
            if pd.notna(entry_rank) and int(entry_rank) > max_rank:
                continue

            regime = str(macro_state.at[current_date, "MacroRegime"])
            if regime == "Risk-off":
                continue

            rs_threshold_spy, rs_threshold_qqq = threshold_map.get(regime, threshold_map["Mixed"])
            stock_20d = stock_change_20d_pct.at[current_date, symbol]
            spy_20d = macro_state.at[current_date, "SPY20dPct"]
            qqq_20d = macro_state.at[current_date, "QQQ20dPct"]
            rs_vs_spy_20d = stock_20d - spy_20d if pd.notna(stock_20d) and pd.notna(spy_20d) else np.nan
            rs_vs_qqq_20d = stock_20d - qqq_20d if pd.notna(stock_20d) and pd.notna(qqq_20d) else np.nan
            if pd.notna(rs_vs_spy_20d) and rs_vs_spy_20d < rs_threshold_spy:
                continue
            if pd.notna(rs_vs_qqq_20d) and rs_vs_qqq_20d < rs_threshold_qqq:
                continue

            if entries_opened_today >= max_new_positions_per_day:
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
            recommendation, reason = build_recommendation(
                review_row,
                bool(leveraged_flags.get(symbol, False)),
                macro_snapshot,
            )
            exit_signal = recommendation == "EXIT"
            if exit_signal:
                position["pending_exit_days"] = int(position.get("pending_exit_days", 0)) + 1
            else:
                position["pending_exit_days"] = 0
            if position["pending_exit_days"] < max(exit_confirm_days, 1):
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

        for symbol in exit_symbols:
            open_positions.pop(symbol, None)

    for symbol, position in open_positions.items():
        last_price = close[symbol].dropna().iloc[-1]
        pnl_pct = ((float(last_price) / float(position["entry_price"])) - 1) * 100
        hold_days = int((close.index[-1] - position["entry_date"]).days)
        trades.append(
            {
                "Symbol": symbol,
                "EntryDate": position["entry_date"].date(),
                "ExitDate": pd.NA,
                "EntrySignalType": position["entry_signal_type"],
                "EntryRank": position["entry_rank"],
                "EntryScore": position["entry_score"],
                "EntryPrice": round(float(position["entry_price"]), 4),
                "ExitPrice": round(float(last_price), 4),
                "PnLPct": round(float(pnl_pct), 2),
                "HoldDays": hold_days,
                "ExitRecommendation": "OPEN",
                "ExitReason": "",
            }
        )

    return pd.DataFrame(trades)


def main() -> None:
    universe = system_signal_backtest.load_universe(str(REPO_ROOT / "dataset" / "combined_universe.csv"))
    leveraged_flags = system_signal_backtest.build_local_leveraged_flags(universe)
    symbols = universe["Symbol"].tolist()

    close, high, low, volume = system_signal_backtest.download_universe_history(symbols)
    usable_symbols = close.columns.tolist()
    leveraged_flags = leveraged_flags.reindex(usable_symbols).fillna(False)
    macro_state = system_signal_backtest.compute_macro_state(close.index)
    state = system_signal_backtest.compute_scanner_state(close, high, low, volume, leveraged_flags, macro_state, trend_mode="ma20")
    exit_state = system_signal_backtest.compute_exit_state(close, volume)

    fixed_current, _ = system_signal_backtest.simulate_trades(
        state,
        exit_state,
        leveraged_flags,
        macro_state,
        system_signal_backtest.DEFAULT_ANALYSIS_DAYS,
        "breakout_only",
        "exit_only",
        0.60,
        -1.0,
        10,
        4,
        3,
        0,
        0.0,
        0.0,
        0.0,
        8.0,
        2.0,
        False,
        None,
    )
    fixed_looser, _ = system_signal_backtest.simulate_trades(
        state,
        exit_state,
        leveraged_flags,
        macro_state,
        system_signal_backtest.DEFAULT_ANALYSIS_DAYS,
        "breakout_only",
        "exit_only",
        0.60,
        -1.0,
        10,
        4,
        3,
        0,
        0.0,
        0.0,
        0.0,
        5.0,
        0.0,
        False,
        None,
    )

    variants = {
        "regime_aware_0m2_50_82": {
            "Risk-on": (0.0, -2.0),
            "Mixed": (5.0, 0.0),
            "Risk-off": (8.0, 2.0),
        },
        "regime_aware_0m2_20_82": {
            "Risk-on": (0.0, -2.0),
            "Mixed": (2.0, 0.0),
            "Risk-off": (8.0, 2.0),
        },
        "regime_aware_0m2_50_50": {
            "Risk-on": (0.0, -2.0),
            "Mixed": (5.0, 0.0),
            "Risk-off": (5.0, 0.0),
        },
    }

    rows = [
        summarize("fixed_8_2", fixed_current),
        summarize("fixed_5_0", fixed_looser),
    ]
    for name, threshold_map in variants.items():
        trades_df = simulate_regime_aware(
            state,
            exit_state,
            leveraged_flags,
            macro_state,
            0.60,
            10,
            4,
            3,
            threshold_map,
        )
        rows.append(summarize(name, trades_df))

    print(pd.DataFrame(rows).sort_values(["avg_ret_pct", "closed_trades"], ascending=[False, False]).to_string(index=False))


if __name__ == "__main__":
    main()
