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

import system_signal_backtest


UNIVERSE_FILE = REPO_ROOT / "dataset" / "major_index_etf.csv"
ENTRY_DEFAULTS = {
    "trend_mode": "ma20",
    "entry_mode": "all_candidates",
    "min_entry_score": 0.55,
    "max_entry_rank": 10,
    "max_new_positions_per_day": 4,
    "min_rs_vs_spy_20d": 0.0,
    "min_rs_vs_qqq_20d": -2.0,
}

EXIT_VARIANTS = [
    {
        "name": "baseline_exit_only_c3",
        "exit_on": "exit_only",
        "exit_confirm_days": 3,
        "profit_lock_trigger_pct": 0.0,
        "profit_lock_drawdown_pct": 0.0,
    },
    {
        "name": "exit_only_c1",
        "exit_on": "exit_only",
        "exit_confirm_days": 1,
        "profit_lock_trigger_pct": 0.0,
        "profit_lock_drawdown_pct": 0.0,
    },
    {
        "name": "exit_only_c2",
        "exit_on": "exit_only",
        "exit_confirm_days": 2,
        "profit_lock_trigger_pct": 0.0,
        "profit_lock_drawdown_pct": 0.0,
    },
    {
        "name": "exit_only_c4",
        "exit_on": "exit_only",
        "exit_confirm_days": 4,
        "profit_lock_trigger_pct": 0.0,
        "profit_lock_drawdown_pct": 0.0,
    },
    {
        "name": "review_or_exit_c1",
        "exit_on": "review_or_exit",
        "exit_confirm_days": 1,
        "profit_lock_trigger_pct": 0.0,
        "profit_lock_drawdown_pct": 0.0,
    },
    {
        "name": "review_or_exit_c2",
        "exit_on": "review_or_exit",
        "exit_confirm_days": 2,
        "profit_lock_trigger_pct": 0.0,
        "profit_lock_drawdown_pct": 0.0,
    },
    {
        "name": "review_or_exit_c3",
        "exit_on": "review_or_exit",
        "exit_confirm_days": 3,
        "profit_lock_trigger_pct": 0.0,
        "profit_lock_drawdown_pct": 0.0,
    },
    {
        "name": "exit_only_c3_lock10_5",
        "exit_on": "exit_only",
        "exit_confirm_days": 3,
        "profit_lock_trigger_pct": 10.0,
        "profit_lock_drawdown_pct": 5.0,
    },
    {
        "name": "exit_only_c3_lock12_6",
        "exit_on": "exit_only",
        "exit_confirm_days": 3,
        "profit_lock_trigger_pct": 12.0,
        "profit_lock_drawdown_pct": 6.0,
    },
    {
        "name": "review_or_exit_c2_lock10_5",
        "exit_on": "review_or_exit",
        "exit_confirm_days": 2,
        "profit_lock_trigger_pct": 10.0,
        "profit_lock_drawdown_pct": 5.0,
    },
]


def summarize(variant: dict[str, object], trades_df: pd.DataFrame) -> dict[str, object]:
    closed = trades_df[trades_df["ExitDate"].notna()].copy() if not trades_df.empty else trades_df
    closed_trades = int(len(closed))
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
        "variant": variant["name"],
        "exit_on": variant["exit_on"],
        "confirm_days": int(variant["exit_confirm_days"]),
        "profit_lock_trigger_pct": float(variant["profit_lock_trigger_pct"]),
        "profit_lock_drawdown_pct": float(variant["profit_lock_drawdown_pct"]),
        "closed_trades": closed_trades,
        "open_trades": int(trades_df["ExitDate"].isna().sum()) if not trades_df.empty else 0,
        "win_rate_pct": round(win_rate, 2) if pd.notna(win_rate) else np.nan,
        "avg_ret_pct": round(avg_ret, 2) if pd.notna(avg_ret) else np.nan,
        "median_ret_pct": round(median_ret, 2) if pd.notna(median_ret) else np.nan,
        "avg_hold_days": round(avg_hold, 1) if pd.notna(avg_hold) else np.nan,
        "annualized_trade_pct": round(annualized, 2) if pd.notna(annualized) else np.nan,
    }


def main() -> None:
    universe = system_signal_backtest.load_universe(str(UNIVERSE_FILE))
    leveraged_flags = system_signal_backtest.build_local_leveraged_flags(universe)
    symbols = universe["Symbol"].tolist()

    close, high, low, volume = system_signal_backtest.download_universe_history(symbols)
    usable_symbols = close.columns.tolist()
    leveraged_flags = leveraged_flags.reindex(usable_symbols).fillna(False)
    macro_state = system_signal_backtest.compute_macro_state(close.index)
    state = system_signal_backtest.compute_scanner_state(
        close,
        high,
        low,
        volume,
        leveraged_flags,
        macro_state,
        trend_mode=str(ENTRY_DEFAULTS["trend_mode"]),
    )
    exit_state = system_signal_backtest.compute_exit_state(close, volume)

    rows: list[dict[str, object]] = []
    for variant in EXIT_VARIANTS:
        trades_df, _ = system_signal_backtest.simulate_trades(
            state,
            exit_state,
            leveraged_flags,
            macro_state,
            system_signal_backtest.DEFAULT_ANALYSIS_DAYS,
            str(ENTRY_DEFAULTS["entry_mode"]),
            str(variant["exit_on"]),
            float(ENTRY_DEFAULTS["min_entry_score"]),
            -1.0,
            int(ENTRY_DEFAULTS["max_entry_rank"]),
            int(ENTRY_DEFAULTS["max_new_positions_per_day"]),
            int(variant["exit_confirm_days"]),
            0,
            0.0,
            float(variant["profit_lock_trigger_pct"]),
            float(variant["profit_lock_drawdown_pct"]),
            float(ENTRY_DEFAULTS["min_rs_vs_spy_20d"]),
            float(ENTRY_DEFAULTS["min_rs_vs_qqq_20d"]),
            False,
            None,
        )
        rows.append(summarize(variant, trades_df))

    results = pd.DataFrame(rows)
    results = results.sort_values(["avg_ret_pct", "closed_trades"], ascending=[False, False]).reset_index(drop=True)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
