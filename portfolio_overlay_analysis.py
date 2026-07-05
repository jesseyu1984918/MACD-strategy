from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MACD_DAILY = "macd_combined_universe_1y_daily.csv"
DEFAULT_BREAKOUT_DAILY = "best_breakout_rank10_portfolio_daily.csv"
DEFAULT_COMBINED_DAILY = "breakout_macd_overlay_daily.csv"
DEFAULT_SUMMARY_OUT = "breakout_macd_overlay_summary.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze lagged portfolio overlays between breakout and MACD daily return streams."
    )
    parser.add_argument("--macd-daily", default=DEFAULT_MACD_DAILY)
    parser.add_argument("--breakout-daily", default=DEFAULT_BREAKOUT_DAILY)
    parser.add_argument("--combined-daily-out", default=DEFAULT_COMBINED_DAILY)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    return parser.parse_args()


def load_daily(path: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    cols = ["Date", "OpenPositions", "Entries", "Exits", "PortfolioDayReturnPct", "Equity"]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df[cols].rename(
        columns={
            "OpenPositions": f"{prefix}_open",
            "Entries": f"{prefix}_entries",
            "Exits": f"{prefix}_exits",
            "PortfolioDayReturnPct": f"{prefix}_ret_pct",
            "Equity": f"{prefix}_eq",
        }
    )


def compute_stats(ret: pd.Series) -> tuple[float, float, float]:
    eq = (1.0 + ret).cumprod()
    total_return_pct = float((eq.iloc[-1] - 1.0) * 100)
    max_drawdown_pct = float(((eq / eq.cummax()) - 1.0).min() * 100)
    sharpe = 0.0
    if ret.std(ddof=0) > 0:
        sharpe = float((ret.mean() / ret.std(ddof=0)) * np.sqrt(252))
    return total_return_pct, max_drawdown_pct, sharpe


def build_overlay_frame(macd_daily: pd.DataFrame, breakout_daily: pd.DataFrame) -> pd.DataFrame:
    df = macd_daily.merge(breakout_daily, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    df["macd_ret"] = df["macd_ret_pct"] / 100.0
    df["breakout_ret"] = df["breakout_ret_pct"] / 100.0
    return df


def evaluate_strategies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, object]] = []

    base_breakout_total, base_breakout_dd, base_breakout_sharpe = compute_stats(df["breakout_ret"])

    breakout_eq_lag = (1.0 + df["breakout_ret"]).cumprod().shift(1).fillna(1.0)
    breakout_dd_lag = (breakout_eq_lag / breakout_eq_lag.cummax()) - 1.0

    relative_strength = {
        "label": "relative_strength_10d_add_35pct_macd",
        "mask": (
            ((1.0 + df["macd_ret"]).rolling(10).apply(np.prod, raw=True) - 1.0).shift(1)
            > ((1.0 + df["breakout_ret"]).rolling(10).apply(np.prod, raw=True) - 1.0).shift(1)
        ),
        "macd_weight": 0.35,
        "family": "relative_strength",
        "description": "Add 35% MACD when MACD trailing 10-day return beat breakout trailing 10-day return as of yesterday.",
    }
    drawdown_overlay = {
        "label": "breakout_drawdown_6pct_add_30pct_macd",
        "mask": breakout_dd_lag < -0.06,
        "macd_weight": 0.30,
        "family": "drawdown",
        "description": "Add 30% MACD when breakout portfolio was more than 6% below its prior peak as of yesterday.",
    }
    capacity_overlay = {
        "label": "breakout_open_lt_10_add_30pct_macd",
        "mask": df["breakout_open"].shift(1).fillna(0) < 10,
        "macd_weight": 0.30,
        "family": "capacity",
        "description": "Add 30% MACD when breakout had fewer than 10 open positions as of yesterday.",
    }

    candidates = [relative_strength, drawdown_overlay, capacity_overlay]
    combined_daily_frames: list[pd.DataFrame] = []

    for candidate in candidates:
        mask = candidate["mask"].fillna(False).astype(bool)
        w = float(candidate["macd_weight"])
        combined_ret = np.where(mask, w * df["macd_ret"] + (1.0 - w) * df["breakout_ret"], df["breakout_ret"])
        combined_ret = pd.Series(combined_ret, index=df.index)
        total, dd, sharpe = compute_stats(combined_ret)
        trigger_days = int(mask.sum())
        combined_eq = (1.0 + combined_ret).cumprod()

        combined_daily = pd.DataFrame(
            {
                "Date": df["Date"].dt.date,
                "OverlayActive": mask,
                "MACDWeight": np.where(mask, w, 0.0),
                "BreakoutWeight": np.where(mask, 1.0 - w, 1.0),
                "MACDReturnPct": df["macd_ret"] * 100.0,
                "BreakoutReturnPct": df["breakout_ret"] * 100.0,
                "CombinedReturnPct": combined_ret * 100.0,
                "CombinedEquity": combined_eq,
                "BreakoutOpenPositions": df["breakout_open"],
                "MACDOpenPositions": df["macd_open"],
            }
        )
        combined_daily["StrategyLabel"] = candidate["label"]
        combined_daily_frames.append(combined_daily)

        results.append(
            {
                "label": candidate["label"],
                "family": candidate["family"],
                "description": candidate["description"],
                "trigger_days": trigger_days,
                "total_return_pct": round(total, 2),
                "max_drawdown_pct": round(dd, 2),
                "sharpe": round(sharpe, 2),
                "improves_return_vs_breakout": total > base_breakout_total,
                "improves_drawdown_vs_breakout": dd > base_breakout_dd,
                "improves_sharpe_vs_breakout": sharpe > base_breakout_sharpe,
            }
        )

    return pd.DataFrame(results), pd.concat(combined_daily_frames, ignore_index=True)


def build_summary(df: pd.DataFrame, results: pd.DataFrame) -> str:
    macd_total, macd_dd, macd_sharpe = compute_stats(df["macd_ret"])
    breakout_total, breakout_dd, breakout_sharpe = compute_stats(df["breakout_ret"])

    recommended = results[
        results["improves_return_vs_breakout"]
        & results["improves_drawdown_vs_breakout"]
        & results["improves_sharpe_vs_breakout"]
    ].sort_values(["sharpe", "total_return_pct"], ascending=[False, False])

    if recommended.empty:
        pick = results.sort_values(["sharpe", "total_return_pct"], ascending=[False, False]).iloc[0]
        recommendation_reason = "No rule improved all three dimensions, so the highest-Sharpe overlay is selected."
    else:
        pick = recommended.iloc[0]
        recommendation_reason = "Selected the highest-Sharpe overlay that also improved return and drawdown versus breakout."

    lines = [
        "Breakout + MACD Overlay Analysis",
        f"Window start: {df['Date'].min().date()}",
        f"Window end: {df['Date'].max().date()}",
        f"Trading days: {len(df)}",
        f"Daily return correlation: {df['macd_ret'].corr(df['breakout_ret']):.4f}",
        "",
        "Base sleeves:",
        f"Breakout total return: {breakout_total:.2f}%",
        f"Breakout max drawdown: {breakout_dd:.2f}%",
        f"Breakout daily Sharpe: {breakout_sharpe:.2f}",
        f"MACD total return: {macd_total:.2f}%",
        f"MACD max drawdown: {macd_dd:.2f}%",
        f"MACD daily Sharpe: {macd_sharpe:.2f}",
        "",
        "Recommended overlay:",
        f"Label: {pick['label']}",
        f"Description: {pick['description']}",
        recommendation_reason,
        f"Trigger days: {int(pick['trigger_days'])}",
        f"Overlay total return: {pick['total_return_pct']:.2f}%",
        f"Overlay max drawdown: {pick['max_drawdown_pct']:.2f}%",
        f"Overlay daily Sharpe: {pick['sharpe']:.2f}",
        "",
        "Evaluated overlays:",
        results[
            [
                "label",
                "trigger_days",
                "total_return_pct",
                "max_drawdown_pct",
                "sharpe",
                "improves_return_vs_breakout",
                "improves_drawdown_vs_breakout",
                "improves_sharpe_vs_breakout",
            ]
        ].to_string(index=False),
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    macd_daily = load_daily(args.macd_daily, "macd")
    breakout_daily = load_daily(args.breakout_daily, "breakout")
    df = build_overlay_frame(macd_daily, breakout_daily)
    results, combined_daily = evaluate_strategies(df)
    summary = build_summary(df, results)

    combined_daily.to_csv(args.combined_daily_out, index=False)
    Path(args.summary_out).write_text(summary)

    print(summary)
    print(f"\nSaved combined daily overlays to {args.combined_daily_out}")
    print(f"Saved summary to {args.summary_out}")


if __name__ == "__main__":
    main()
