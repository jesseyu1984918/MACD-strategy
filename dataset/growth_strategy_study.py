from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from yfinance import cache as yf_cache

import system_signal_backtest


DATASET_DIR = Path(__file__).resolve().parent
UNIVERSE_FILE = DATASET_DIR / "growth.csv"
RESULTS_FILE = DATASET_DIR / "growth_study_results.csv"
REGRESSION_FILE = DATASET_DIR / "growth_regression_coefficients.csv"
BEST_TRADES_FILE = DATASET_DIR / "growth_best_trades.csv"
BEST_DAILY_FILE = DATASET_DIR / "growth_best_daily.csv"
BEST_SUMMARY_FILE = DATASET_DIR / "growth_best_summary.txt"
YF_CACHE_DIR = DATASET_DIR / ".yf_cache"


PARAM_GRID = {
    "trend_mode": ["ma20", "ma50", "hybrid"],
    "entry_mode": ["breakout_only"],
    "exit_on": ["exit_only"],
    "min_entry_score": [0.60, 0.62, 0.64, 0.66],
    "leader_min_entry_score": [-1.0, 0.70],
    "max_entry_rank": [8, 10],
    "max_new_positions_per_day": [3, 4],
    "exit_confirm_days": [3, 4],
    "min_rs_vs_spy_20d": [5.0, 8.0],
    "min_rs_vs_qqq_20d": [0.0],
    "allow_risk_off_entries": [False],
}


def summarize_run(params: dict[str, object], trades_df: pd.DataFrame) -> dict[str, object]:
    closed = trades_df[trades_df["ExitRecommendation"].ne("OPEN")].copy() if not trades_df.empty else trades_df
    open_trades = trades_df[trades_df["ExitRecommendation"].eq("OPEN")].copy() if not trades_df.empty else trades_df
    return {
        **params,
        "closed_trades": int(len(closed)),
        "open_trades": int(len(open_trades)),
        "win_rate_pct": round(float((closed["PnLPct"] > 0).mean() * 100), 4) if not closed.empty else 0.0,
        "avg_return_pct": round(float(closed["PnLPct"].mean()), 4) if not closed.empty else 0.0,
        "median_return_pct": round(float(closed["PnLPct"].median()), 4) if not closed.empty else 0.0,
        "total_return_pct_sum": round(float(closed["PnLPct"].sum()), 4) if not closed.empty else 0.0,
        "avg_hold_days": round(float(closed["HoldDays"].mean()), 4) if not closed.empty else 0.0,
        "avg_entry_rank": round(float(closed["EntryRank"].mean()), 4) if not closed.empty else 0.0,
        "max_gain_pct": round(float(closed["PnLPct"].max()), 4) if not closed.empty else 0.0,
        "max_loss_pct": round(float(closed["PnLPct"].min()), 4) if not closed.empty else 0.0,
    }


def fit_regression(results_df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "trend_mode",
        "entry_mode",
        "exit_on",
        "min_entry_score",
        "leader_min_entry_score",
        "max_entry_rank",
        "max_new_positions_per_day",
        "exit_confirm_days",
        "min_rs_vs_spy_20d",
        "min_rs_vs_qqq_20d",
        "allow_risk_off_entries",
    ]
    categorical = ["trend_mode", "entry_mode", "exit_on", "allow_risk_off_entries"]
    numeric = [col for col in feature_columns if col not in categorical]

    model = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("num", "passthrough", numeric),
                    ]
                ),
            ),
            ("model", LinearRegression()),
        ]
    )

    x = results_df[feature_columns]
    y = results_df["total_return_pct_sum"]
    model.fit(x, y)

    prep = model.named_steps["prep"]
    linear = model.named_steps["model"]
    feature_names = prep.get_feature_names_out()
    coefs = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": linear.coef_,
        }
    ).sort_values("coefficient", ascending=False, key=lambda s: s.abs()).reset_index(drop=True)
    coefs["intercept"] = float(linear.intercept_)
    coefs["r2"] = float(model.score(x, y))
    return coefs


def build_best_summary(best_row: pd.Series, regression_df: pd.DataFrame, variant_count: int) -> str:
    positive = regression_df.sort_values("coefficient", ascending=False).head(8)
    negative = regression_df.sort_values("coefficient", ascending=True).head(8)
    lines = [
        "Growth Strategy Study",
        f"Universe: {UNIVERSE_FILE.name}",
        f"Variants tested: {variant_count}",
        "",
        "Best run by total closed-trade return sum:",
        f"trend_mode={best_row['trend_mode']}",
        f"entry_mode={best_row['entry_mode']}",
        f"exit_on={best_row['exit_on']}",
        f"min_entry_score={best_row['min_entry_score']}",
        f"leader_min_entry_score={best_row['leader_min_entry_score']}",
        f"max_entry_rank={int(best_row['max_entry_rank'])}",
        f"max_new_positions_per_day={int(best_row['max_new_positions_per_day'])}",
        f"exit_confirm_days={int(best_row['exit_confirm_days'])}",
        f"min_rs_vs_spy_20d={best_row['min_rs_vs_spy_20d']}",
        f"min_rs_vs_qqq_20d={best_row['min_rs_vs_qqq_20d']}",
        f"closed_trades={int(best_row['closed_trades'])}",
        f"open_trades={int(best_row['open_trades'])}",
        f"win_rate_pct={best_row['win_rate_pct']}",
        f"avg_return_pct={best_row['avg_return_pct']}",
        f"median_return_pct={best_row['median_return_pct']}",
        f"total_return_pct_sum={best_row['total_return_pct_sum']}",
        f"avg_hold_days={best_row['avg_hold_days']}",
        f"max_gain_pct={best_row['max_gain_pct']}",
        f"max_loss_pct={best_row['max_loss_pct']}",
        "",
        f"Regression R^2 on tested grid: {regression_df['r2'].iloc[0]:.4f}",
        f"Regression intercept: {regression_df['intercept'].iloc[0]:.4f}",
        "",
        "Most positive parameter effects on total_return_pct_sum:",
    ]
    lines.extend(f"{row.feature}: {row.coefficient:.4f}" for row in positive.itertuples(index=False))
    lines.append("")
    lines.append("Most negative parameter effects on total_return_pct_sum:")
    lines.extend(f"{row.feature}: {row.coefficient:.4f}" for row in negative.itertuples(index=False))
    return "\n".join(lines)


def main() -> None:
    YF_CACHE_DIR.mkdir(exist_ok=True)
    yf_cache.set_cache_location(str(YF_CACHE_DIR))

    universe = system_signal_backtest.load_universe(str(UNIVERSE_FILE))
    leveraged_flags = system_signal_backtest.build_local_leveraged_flags(universe)
    symbols = universe["Symbol"].tolist()

    close, high, low, volume = system_signal_backtest.download_universe_history(symbols)
    usable_symbols = close.columns.tolist()
    leveraged_flags = leveraged_flags.reindex(usable_symbols).fillna(False)
    macro_state = system_signal_backtest.compute_macro_state(close.index)
    exit_state = system_signal_backtest.compute_exit_state(close, volume)
    state_cache = {
        trend_mode: system_signal_backtest.compute_scanner_state(
            close,
            high,
            low,
            volume,
            leveraged_flags,
            macro_state,
            trend_mode=trend_mode,
        )
        for trend_mode in PARAM_GRID["trend_mode"]
    }

    results: list[dict[str, object]] = []
    run_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    keys = list(PARAM_GRID.keys())
    variants = list(product(*(PARAM_GRID[key] for key in keys)))
    for index, values in enumerate(variants, start=1):
        params = dict(zip(keys, values))
        state = state_cache[str(params["trend_mode"])]
        trades_df, daily_df = system_signal_backtest.simulate_trades(
            state,
            exit_state,
            leveraged_flags,
            macro_state,
            system_signal_backtest.DEFAULT_ANALYSIS_DAYS,
            str(params["entry_mode"]),
            str(params["exit_on"]),
            float(params["min_entry_score"]),
            float(params["leader_min_entry_score"]),
            int(params["max_entry_rank"]),
            int(params["max_new_positions_per_day"]),
            int(params["exit_confirm_days"]),
            0,
            0.0,
            0.0,
            0.0,
            float(params["min_rs_vs_spy_20d"]),
            float(params["min_rs_vs_qqq_20d"]),
            bool(params["allow_risk_off_entries"]),
            None,
        )
        key = "|".join(str(params[name]) for name in keys)
        run_cache[key] = (trades_df, daily_df)
        results.append(summarize_run(params, trades_df))
        if index % 25 == 0 or index == len(variants):
            print(f"Completed {index} of {len(variants)} variants")

    results_df = pd.DataFrame(results).sort_values(
        ["total_return_pct_sum", "avg_return_pct", "win_rate_pct", "closed_trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    results_df.to_csv(RESULTS_FILE, index=False)

    regression_df = fit_regression(results_df)
    regression_df.to_csv(REGRESSION_FILE, index=False)

    best_row = results_df.iloc[0]
    best_key = "|".join(str(best_row[name]) for name in keys)
    best_trades_df, best_daily_df = run_cache[best_key]
    best_trades_df.to_csv(BEST_TRADES_FILE, index=False)
    best_daily_df.to_csv(BEST_DAILY_FILE, index=False)
    BEST_SUMMARY_FILE.write_text(build_best_summary(best_row, regression_df, len(results_df)))

    print(BEST_SUMMARY_FILE.read_text())
    print(f"\nSaved sweep results to {RESULTS_FILE}")
    print(f"Saved regression coefficients to {REGRESSION_FILE}")
    print(f"Saved best trades to {BEST_TRADES_FILE}")
    print(f"Saved best daily signals to {BEST_DAILY_FILE}")
    print(f"Saved best summary to {BEST_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
