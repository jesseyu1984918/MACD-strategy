from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
import sys

VENV_SITE_PACKAGES = Path(__file__).resolve().parent / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import numpy as np
import pandas as pd
import yfinance as yf


TRADE_TYPES = {"Bought", "Sold", "Sold Short", "Bought To Cover"}
OPTION_PATTERN = r"CALL|PUT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose trading history with historical price context.")
    parser.add_argument("csv_path", nargs="?", default="DownloadTxnHistory.csv")
    parser.add_argument("--trades-out", default="trade_diagnosis.csv")
    parser.add_argument("--summary-out", default="trade_diagnosis_summary.txt")
    return parser.parse_args()


def load_transactions(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    lines = path.read_text().splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Activity/Trade Date,"))

    df = pd.read_csv(path, skiprows=header_idx)
    df = df[pd.to_datetime(df["Activity/Trade Date"], format="%m/%d/%y", errors="coerce").notna()].copy()
    df["trade_date"] = pd.to_datetime(df["Activity/Trade Date"], format="%m/%d/%y")

    for col in ["Quantity #", "Price $", "Amount $", "Commission"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Activity Type"].isin(TRADE_TYPES)].copy()
    df = df[~df["Description"].fillna("").str.contains(OPTION_PATTERN, case=False, regex=True)].copy()
    df = df[df["Symbol"].notna() & (df["Symbol"] != "--")].copy()
    df = df.sort_values(["trade_date", "Transaction Date", "Settlement Date"]).reset_index(drop=True)
    return df


def reconstruct_round_trips(df: pd.DataFrame) -> pd.DataFrame:
    long_lots: dict[str, deque] = defaultdict(deque)
    short_lots: dict[str, deque] = defaultdict(deque)
    closed: list[dict] = []

    for _, row in df.iterrows():
        symbol = row["Symbol"]
        quantity = float(row["Quantity #"]) if pd.notna(row["Quantity #"]) else 0.0
        price = float(row["Price $"]) if pd.notna(row["Price $"]) else 0.0
        trade_date = row["trade_date"]
        activity = row["Activity Type"]

        if activity == "Bought":
            remaining = quantity
            while remaining > 1e-9 and short_lots[symbol]:
                lot = short_lots[symbol][0]
                matched = min(remaining, lot["qty"])
                pnl = (lot["entry_price"] - price) * matched
                closed.append(
                    build_closed_trade(
                        symbol=symbol,
                        side="SHORT",
                        entry_date=lot["entry_date"],
                        exit_date=trade_date,
                        entry_price=lot["entry_price"],
                        exit_price=price,
                        quantity=matched,
                        pnl=pnl,
                    )
                )
                lot["qty"] -= matched
                remaining -= matched
                if lot["qty"] <= 1e-9:
                    short_lots[symbol].popleft()

            if remaining > 1e-9:
                long_lots[symbol].append({"qty": remaining, "entry_price": price, "entry_date": trade_date})

        elif activity == "Sold":
            remaining = abs(quantity)
            while remaining > 1e-9 and long_lots[symbol]:
                lot = long_lots[symbol][0]
                matched = min(remaining, lot["qty"])
                pnl = (price - lot["entry_price"]) * matched
                closed.append(
                    build_closed_trade(
                        symbol=symbol,
                        side="LONG",
                        entry_date=lot["entry_date"],
                        exit_date=trade_date,
                        entry_price=lot["entry_price"],
                        exit_price=price,
                        quantity=matched,
                        pnl=pnl,
                    )
                )
                lot["qty"] -= matched
                remaining -= matched
                if lot["qty"] <= 1e-9:
                    long_lots[symbol].popleft()

        elif activity == "Sold Short":
            short_lots[symbol].append({"qty": abs(quantity), "entry_price": price, "entry_date": trade_date})

        elif activity == "Bought To Cover":
            remaining = quantity
            while remaining > 1e-9 and short_lots[symbol]:
                lot = short_lots[symbol][0]
                matched = min(remaining, lot["qty"])
                pnl = (lot["entry_price"] - price) * matched
                closed.append(
                    build_closed_trade(
                        symbol=symbol,
                        side="SHORT",
                        entry_date=lot["entry_date"],
                        exit_date=trade_date,
                        entry_price=lot["entry_price"],
                        exit_price=price,
                        quantity=matched,
                        pnl=pnl,
                    )
                )
                lot["qty"] -= matched
                remaining -= matched
                if lot["qty"] <= 1e-9:
                    short_lots[symbol].popleft()

    closed_df = pd.DataFrame(closed)
    closed_df["return_pct"] = np.where(
        closed_df["side"] == "LONG",
        (closed_df["exit_price"] / closed_df["entry_price"] - 1) * 100,
        (closed_df["entry_price"] / closed_df["exit_price"] - 1) * 100,
    )
    return closed_df.sort_values(["entry_date", "exit_date", "symbol"]).reset_index(drop=True)


def build_closed_trade(
    *,
    symbol: str,
    side: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    quantity: float,
    pnl: float,
) -> dict:
    return {
        "symbol": symbol,
        "side": side,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "notional": entry_price * quantity,
        "pnl": pnl,
        "hold_days": int((exit_date - entry_date).days),
    }


def flatten_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_history(
    symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    history: dict[str, pd.DataFrame] = {}
    splits_map: dict[str, pd.Series] = {}

    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        hist = yf.download(
            symbol,
            start=(start - pd.Timedelta(days=250)).strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )
        hist = flatten_history_columns(hist)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist = hist.rename(columns=str.title)
        hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        history[symbol] = add_indicators(hist)
        splits = ticker.splits
        if splits is None or len(splits) == 0:
            splits_map[symbol] = pd.Series(dtype=float)
        else:
            splits.index = pd.to_datetime(splits.index).tz_localize(None)
            splits_map[symbol] = splits.astype(float)

    return history, splits_map


def add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    data = hist.copy()
    data["ma20"] = data["Close"].rolling(20).mean()
    data["ma50"] = data["Close"].rolling(50).mean()
    data["hh50"] = data["Close"].rolling(50).max()
    data["runup_10d_pct"] = data["Close"].pct_change(10) * 100

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi14"] = 100 - 100 / (1 + rs)

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    data["macd_hist"] = macd - signal
    return data


def get_snapshot(hist: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    eligible = hist.loc[hist.index <= date]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def split_adjusted_price(raw_price: float, trade_date: pd.Timestamp, splits: pd.Series) -> float:
    if splits.empty:
        return raw_price
    future_splits = splits.loc[splits.index > trade_date]
    if future_splits.empty:
        return raw_price
    return raw_price / future_splits.product()


def enrich_closed_trades(
    closed_df: pd.DataFrame, history: dict[str, pd.DataFrame], splits_map: dict[str, pd.Series]
) -> pd.DataFrame:
    records = []

    for row in closed_df.itertuples(index=False):
        hist = history.get(row.symbol)
        splits = splits_map.get(row.symbol, pd.Series(dtype=float))
        if hist is None or hist.empty:
            continue

        entry = get_snapshot(hist, row.entry_date)
        exit_ = get_snapshot(hist, row.exit_date)
        if entry is None or exit_ is None:
            continue

        entry_price_adj = split_adjusted_price(row.entry_price, row.entry_date, splits)
        exit_price_adj = split_adjusted_price(row.exit_price, row.exit_date, splits)
        hold_window = hist.loc[(hist.index >= row.entry_date) & (hist.index <= row.exit_date)]
        if hold_window.empty:
            hold_window = hist.loc[[entry.name]]

        if row.side == "LONG":
            mfe_pct = (hold_window["High"].max() / entry_price_adj - 1) * 100
            mae_pct = (hold_window["Low"].min() / entry_price_adj - 1) * 100
        else:
            mfe_pct = (1 - hold_window["Low"].min() / entry_price_adj) * 100
            mae_pct = (1 - hold_window["High"].max() / entry_price_adj) * 100

        records.append(
            {
                **row._asdict(),
                "entry_price_adj": entry_price_adj,
                "exit_price_adj": exit_price_adj,
                "entry_above_ma20_pct": pct_from_level(entry_price_adj, entry.get("ma20")),
                "entry_above_ma50_pct": pct_from_level(entry_price_adj, entry.get("ma50")),
                "entry_from_50d_high_pct": pct_from_level(entry_price_adj, entry.get("hh50")),
                "entry_10d_runup_pct": entry.get("runup_10d_pct"),
                "entry_rsi14": entry.get("rsi14"),
                "entry_macd_hist": entry.get("macd_hist"),
                "exit_above_ma20_pct": pct_from_level(exit_price_adj, exit_.get("ma20")),
                "exit_above_ma50_pct": pct_from_level(exit_price_adj, exit_.get("ma50")),
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
            }
        )

    return pd.DataFrame(records)


def pct_from_level(price: float, level: float | None) -> float:
    if level is None or pd.isna(level) or level == 0:
        return np.nan
    return (price / level - 1) * 100


def summarize(enriched: pd.DataFrame) -> str:
    lines: list[str] = []

    total_pnl = enriched["pnl"].sum()
    win_rate = (enriched["pnl"] > 0).mean() * 100
    avg_hold = enriched["hold_days"].mean()
    avg_winner_hold = enriched.loc[enriched["pnl"] > 0, "hold_days"].mean()
    avg_loser_hold = enriched.loc[enriched["pnl"] <= 0, "hold_days"].mean()

    lines.append("Trade Diagnosis")
    lines.append(f"Closed trades analyzed: {len(enriched)}")
    lines.append(f"Net realized PnL from matched round trips: {total_pnl:,.2f}")
    lines.append(f"Win rate: {win_rate:.1f}%")
    lines.append(f"Average hold: {avg_hold:.1f} days")
    lines.append(f"Average winner hold: {avg_winner_hold:.1f} days")
    lines.append(f"Average loser hold: {avg_loser_hold:.1f} days")
    lines.append("")

    long_df = enriched[enriched["side"] == "LONG"]
    short_df = enriched[enriched["side"] == "SHORT"]

    chase_longs = long_df[long_df["entry_above_ma20_pct"] > 8]
    runup_longs = long_df[long_df["entry_10d_runup_pct"] > 15]
    weak_trend_longs = long_df[long_df["entry_above_ma50_pct"] < 0]
    uptrend_shorts = short_df[(short_df["entry_above_ma20_pct"] > 0) & (short_df["entry_above_ma50_pct"] > 0)]

    lines.append("Behavior Flags")
    lines.append(
        f"Long entries >8% above the 20-day average: {len(chase_longs)} trades, "
        f"PnL {chase_longs['pnl'].sum():,.2f}"
    )
    lines.append(
        f"Long entries after >15% 10-day run-up: {len(runup_longs)} trades, "
        f"PnL {runup_longs['pnl'].sum():,.2f}"
    )
    lines.append(
        f"Long entries below the 50-day average: {len(weak_trend_longs)} trades, "
        f"PnL {weak_trend_longs['pnl'].sum():,.2f}"
    )
    lines.append(
        f"Short entries placed above both the 20-day and 50-day averages: {len(uptrend_shorts)} trades, "
        f"PnL {uptrend_shorts['pnl'].sum():,.2f}"
    )
    lines.append("")

    lines.append("Largest Loss Buckets")
    worst_symbols = enriched.groupby("symbol")["pnl"].sum().sort_values().head(10)
    for symbol, pnl in worst_symbols.items():
        lines.append(f"{symbol}: {pnl:,.2f}")
    lines.append("")

    lines.append("Largest Individual Losses")
    cols = ["symbol", "side", "entry_date", "exit_date", "pnl", "return_pct", "entry_above_ma20_pct", "entry_10d_runup_pct"]
    for row in enriched.sort_values("pnl").head(10)[cols].itertuples(index=False):
        lines.append(
            f"{row.symbol} {row.side} {row.entry_date.date()}->{row.exit_date.date()} "
            f"PnL {row.pnl:,.2f} Return {row.return_pct:.1f}% "
            f"EntryVsMA20 {row.entry_above_ma20_pct:.1f}% Runup10d {row.entry_10d_runup_pct:.1f}%"
        )
    lines.append("")

    lines.append("Interpretation")
    if avg_loser_hold > avg_winner_hold:
        lines.append("Losers were held longer than winners. Exit discipline is hurting more than raw entry selection.")
    if len(chase_longs) and chase_longs["pnl"].sum() < 0:
        lines.append("A meaningful share of losses came from buying names already extended above the 20-day average.")
    if len(runup_longs) and runup_longs["pnl"].sum() < 0:
        lines.append("Several losing longs were initiated after sharp 10-day runs, which is consistent with chasing.")
    if len(uptrend_shorts) and uptrend_shorts["pnl"].sum() < 0:
        lines.append("Shorts were often opened into still-strong uptrends instead of actual trend breakdowns.")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    txns = load_transactions(args.csv_path)
    closed = reconstruct_round_trips(txns)

    start = closed["entry_date"].min()
    end = closed["exit_date"].max()
    history, splits_map = download_history(sorted(closed["symbol"].unique()), start, end)
    enriched = enrich_closed_trades(closed, history, splits_map)

    enriched = enriched.sort_values(["entry_date", "exit_date", "symbol"]).reset_index(drop=True)
    enriched.to_csv(args.trades_out, index=False)

    summary = summarize(enriched)
    Path(args.summary_out).write_text(summary)
    print(summary)
    print("")
    print(f"Saved detailed trade file to {args.trades_out}")
    print(f"Saved summary to {args.summary_out}")


if __name__ == "__main__":
    main()
