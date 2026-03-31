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

from macro_market_status import get_macro_regime_snapshot
from news_sentiment import get_symbol_news_sentiment
from scanner import MAX_BREAKOUT_AGE, MAX_BREAKOUT_EXTENSION, SETUP_PROXIMITY, bars_since_true
from trade_guardrails import MAX_ABOVE_MA20_PCT, build_metadata_flags, flatten_columns


TRADE_TYPES = {"Bought", "Sold", "Sold Short", "Bought To Cover"}
OPTION_PATTERN = r"CALL|PUT"
OUTPUT_FILE = "position_exit_review.csv"
# Calibrated from historical scanner/backtest score distribution:
# about the 20th percentile for weak holds and the 50th percentile for solid holds.
EXIT_REVIEW_SCORE_REVIEW_THRESHOLD = 0.40
EXIT_REVIEW_SCORE_HOLD_THRESHOLD = 0.56


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review current open positions and suggest hold/review/exit.")
    parser.add_argument("csv_path", nargs="?", default="DownloadTxnHistory.csv")
    parser.add_argument("--output", default=OUTPUT_FILE)
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
    return df.sort_values(["trade_date", "Transaction Date", "Settlement Date"]).reset_index(drop=True)


def load_simple_positions(csv_path: str) -> pd.DataFrame:
    lines = Path(csv_path).read_text().splitlines()

    if lines and " Positions, as of " in lines[0]:
        df = pd.read_csv(csv_path, skiprows=1)
        rows = []
        pending_option_symbol = None
        pending_option_type = None

        for _, row in df.iterrows():
            position_text = str(row.get("Position", "")).strip()
            if not position_text or position_text.startswith("Portfolio Analysis") or position_text.startswith("Cash on Deposit"):
                continue

            if pending_option_symbol and position_text.startswith("+"):
                detail_text = position_text.replace("+", "").strip()
                detail_parts = detail_text.split()
                expiration = detail_parts[1] if len(detail_parts) > 1 else None
                strike = pd.to_numeric(detail_parts[2].replace("$", ""), errors="coerce") if len(detail_parts) > 2 else np.nan

                rows.append(
                    {
                        "Symbol": pending_option_symbol,
                        "Side": "LONG",
                        "Quantity": pd.to_numeric(row.get("Qty"), errors="coerce"),
                        "CostBasis": pd.to_numeric(row.get("Net Cost Basis"), errors="coerce"),
                        "FirstEntryDate": pd.NaT,
                        "InstrumentType": pending_option_type,
                        "UnderlyingPrice": pd.to_numeric(row.get("U/L Price"), errors="coerce"),
                        "OptionMark": pd.to_numeric(row.get("Mark"), errors="coerce"),
                        "EarningsDate": row.get("Earnings Date"),
                        "Expiration": expiration,
                        "Strike": strike,
                    }
                )
                pending_option_symbol = None
                pending_option_type = None
                continue

            parts = position_text.split()
            if not parts:
                continue

            symbol = parts[0].upper()
            lower_position = position_text.lower()
            if " call" in lower_position:
                pending_option_symbol = symbol
                pending_option_type = "CALL"
                continue
            elif " put" in lower_position:
                pending_option_symbol = symbol
                pending_option_type = "PUT"
                continue

            rows.append(
                {
                    "Symbol": symbol,
                    "Side": "LONG",
                    "Quantity": pd.to_numeric(row.get("Qty"), errors="coerce"),
                    "CostBasis": pd.to_numeric(row.get("Net Cost Basis"), errors="coerce"),
                    "FirstEntryDate": pd.NaT,
                    "InstrumentType": "EQUITY",
                    "UnderlyingPrice": pd.to_numeric(row.get("U/L Price"), errors="coerce"),
                    "OptionMark": pd.to_numeric(row.get("Mark"), errors="coerce"),
                    "EarningsDate": row.get("Earnings Date"),
                    "Expiration": None,
                    "Strike": np.nan,
                }
            )

        return pd.DataFrame(rows).reset_index(drop=True)

    df = pd.read_csv(csv_path)
    normalized = {str(col).strip().lower().replace(" ", "").replace("_", ""): col for col in df.columns}

    symbol_col = normalized.get("symbol") or normalized.get("ticker")
    if symbol_col is None:
        first_col = df.columns[0]
        symbol_col = first_col

    cost_col = (
        normalized.get("costbasis")
        or normalized.get("avgcost")
        or normalized.get("averagecost")
        or normalized.get("cost")
    )
    qty_col = (
        normalized.get("quantity")
        or normalized.get("shares")
        or normalized.get("share")
        or normalized.get("volume")
    )
    side_col = normalized.get("side") or normalized.get("positiontype")

    positions = pd.DataFrame()
    positions["Symbol"] = df[symbol_col].dropna().astype(str).str.strip()
    positions = positions[positions["Symbol"].str.lower().ne("symbol")].copy()
    positions["Side"] = df.loc[positions.index, side_col].astype(str).str.upper().str.strip() if side_col else "LONG"
    positions["Side"] = positions["Side"].replace({"": "LONG", "BUY": "LONG", "SELLSHORT": "SHORT"})
    positions["Quantity"] = pd.to_numeric(df.loc[positions.index, qty_col], errors="coerce") if qty_col else np.nan
    positions["CostBasis"] = pd.to_numeric(df.loc[positions.index, cost_col], errors="coerce") if cost_col else np.nan
    positions["FirstEntryDate"] = pd.NaT
    positions["InstrumentType"] = "EQUITY"
    positions["UnderlyingPrice"] = np.nan
    positions["OptionMark"] = np.nan
    positions["EarningsDate"] = None
    positions["Expiration"] = None
    positions["Strike"] = np.nan
    return positions.reset_index(drop=True)


def reconstruct_open_positions(df: pd.DataFrame) -> pd.DataFrame:
    long_lots: dict[str, deque] = defaultdict(deque)
    short_lots: dict[str, deque] = defaultdict(deque)

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
                lot["qty"] -= matched
                remaining -= matched
                if lot["qty"] <= 1e-9:
                    short_lots[symbol].popleft()

    open_rows = []
    for symbol, lots in long_lots.items():
        total_qty = sum(lot["qty"] for lot in lots)
        if total_qty > 1e-9:
            weighted_cost = sum(lot["qty"] * lot["entry_price"] for lot in lots) / total_qty
            first_entry = min(lot["entry_date"] for lot in lots)
            open_rows.append(
                {
                    "Symbol": symbol,
                    "Side": "LONG",
                    "Quantity": total_qty,
                    "CostBasis": weighted_cost,
                    "FirstEntryDate": first_entry,
                }
            )

    for symbol, lots in short_lots.items():
        total_qty = sum(lot["qty"] for lot in lots)
        if total_qty > 1e-9:
            weighted_cost = sum(lot["qty"] * lot["entry_price"] for lot in lots) / total_qty
            first_entry = min(lot["entry_date"] for lot in lots)
            open_rows.append(
                {
                    "Symbol": symbol,
                    "Side": "SHORT",
                    "Quantity": total_qty,
                    "CostBasis": weighted_cost,
                    "FirstEntryDate": first_entry,
                }
            )

    return pd.DataFrame(open_rows).sort_values(["Side", "Symbol"]).reset_index(drop=True)


def download_history(symbol: str) -> pd.DataFrame | None:
    try:
        hist = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
        hist = flatten_columns(hist)
        if hist is None or hist.empty:
            return None
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist = hist.rename(columns=str.title)
        hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if hist.empty:
            return None

        hist["ma20"] = hist["Close"].rolling(20).mean()
        hist["ma50"] = hist["Close"].rolling(50).mean()
        hist["ma150"] = hist["Close"].rolling(150).mean()
        hist["runup_10d_pct"] = hist["Close"].pct_change(10) * 100
        hist["avg_dollar_vol_20d"] = (hist["Close"] * hist["Volume"]).rolling(20).mean()
        hist["ma20_slope"] = hist["ma20"].diff()

        ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
        ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist["macd_hist"] = macd - signal

        hh50 = hist["Close"].rolling(50).max()
        hist["prior_hh50"] = hh50.shift(1)
        breakout = hist["High"] >= hist["prior_hh50"]
        hist["breakout_age"] = bars_since_true(pd.DataFrame({"signal": breakout}))["signal"]
        hist["extension_from_ma20"] = hist["Close"] / hist["ma20"] - 1
        hist["extension_from_breakout"] = hist["Close"] / hist["prior_hh50"] - 1
        hist["scanner_trend"] = (hist["Close"] > hist["ma50"]) & (hist["ma50"] > hist["ma150"])
        hist["close_to_breakout"] = hist["Close"] >= hist["prior_hh50"] * (1 - SETUP_PROXIMITY)
        hist["pre_breakout"] = hist["Close"] < hist["prior_hh50"]
        hist["fresh_breakout"] = hist["breakout_age"] <= MAX_BREAKOUT_AGE
        hist["not_extended"] = hist["extension_from_ma20"] <= 0.08
        hist["pullback_zone"] = hist["extension_from_ma20"] >= -0.02
        hist["controlled_breakout"] = hist["extension_from_breakout"] <= MAX_BREAKOUT_EXTENSION
        hist["setup_candidate"] = (
            hist["scanner_trend"]
            & hist["close_to_breakout"]
            & hist["pre_breakout"]
            & hist["not_extended"]
            & hist["pullback_zone"]
        )
        hist["breakout_candidate"] = (
            hist["scanner_trend"]
            & hist["fresh_breakout"]
            & hist["controlled_breakout"]
            & hist["not_extended"]
            & hist["pullback_zone"]
        )

        prev_close = hist["Close"].shift(1)
        tr_components = np.column_stack(
            [
                (hist["High"] - hist["Low"]).to_numpy(),
                (hist["High"] - prev_close).abs().to_numpy(),
                (hist["Low"] - prev_close).abs().to_numpy(),
            ]
        )
        hist["tr"] = np.nanmax(tr_components, axis=1)
        hist["atr20"] = hist["tr"].rolling(20).mean()
        hist["atr200"] = hist["tr"].rolling(200).mean()
        hist["tightness"] = (1 - hist["atr20"] / hist["atr200"]).clip(0, 1)
        hist["setup_readiness"] = (
            1 - (((hist["prior_hh50"] - hist["Close"]) / hist["prior_hh50"]).clip(lower=0) / SETUP_PROXIMITY).clip(lower=0, upper=1)
        )
        hist["trend_strength"] = ((hist["ma50"] / hist["ma150"]) - 1).clip(lower=0, upper=0.20) / 0.20
        hist["ma_stretch_score"] = 1 - (hist["extension_from_ma20"].abs() / (MAX_ABOVE_MA20_PCT / 100)).clip(lower=0, upper=1)
        hist["breakout_extension_score"] = (
            1 - (hist["extension_from_breakout"].clip(lower=0) / MAX_BREAKOUT_EXTENSION).clip(lower=0, upper=1)
        )
        hist["freshness_score"] = (1 - (hist["breakout_age"] / MAX_BREAKOUT_AGE).clip(lower=0, upper=1)).fillna(0)
        hist["signal_type_bonus"] = hist["setup_candidate"].astype(float) * 1.0 + hist["breakout_candidate"].astype(float) * 0.7
        hist["scanner_score"] = (
            0.28 * hist["setup_readiness"]
            + 0.20 * hist["tightness"]
            + 0.18 * hist["trend_strength"]
            + 0.16 * hist["ma_stretch_score"]
            + 0.12 * hist["breakout_extension_score"]
            + 0.06 * hist["freshness_score"]
            + 0.06 * hist["signal_type_bonus"]
        )
        return hist
    except Exception:
        return None


def assess_exit_state(
    row: pd.Series,
    leveraged_flag: bool,
    macro_snapshot: dict[str, object],
) -> dict[str, object]:
    reasons: list[str] = []
    exit_flags = 0
    review_flags = 0
    profit_protection_triggered = False
    threshold_shift = float(macro_snapshot["exit_review_threshold_shift"])
    macro_regime = str(macro_snapshot["regime"])
    review_threshold = EXIT_REVIEW_SCORE_REVIEW_THRESHOLD + threshold_shift
    hold_threshold = EXIT_REVIEW_SCORE_HOLD_THRESHOLD + threshold_shift

    if leveraged_flag:
        review_flags += 1
        reasons.append("leveraged_or_inverse_product")

    if row["Side"] == "LONG":
        structural_breakdown = (
            row["LastPrice"] < row["MA50"]
            and row["LastPrice"] < row["MA20"]
            and row["MA20Slope"] < 0
        )
        scanner_signal_type = row.get("ScannerSignalType", "")
        breakout_age = row.get("BreakoutAge", np.nan)
        dist_from_breakout_pct = row.get("DistFromBreakoutPct", np.nan)
        scanner_score = row.get("ScannerScore", np.nan)

        has_cost_basis = pd.notna(row["PnLPct"])

        if macro_regime == "Risk-off" and scanner_signal_type in {"SETUP", "BREAKOUT"}:
            review_flags += 1
            reasons.append("macro_risk_off_backdrop")

        if structural_breakdown and ((has_cost_basis and row["PnLPct"] < 0) or row["LastPrice"] < row["MA150"]):
            exit_flags += 1
            reasons.append("below_ma20_and_ma50_with_falling_ma20")
        if row["MACDHist"] < 0 and row["LastPrice"] < row["MA20"] and ((has_cost_basis and row["PnLPct"] < 0) or row["LastPrice"] < row["MA50"]):
            exit_flags += 1
            reasons.append("negative_macd_below_ma20")
        if has_cost_basis and row["PnLPct"] < -8:
            exit_flags += 1
            reasons.append("loss_exceeds_8pct")
        if has_cost_basis and row["PnLPct"] >= 10 and row["LastPrice"] < row["MA20"] and row["MACDHist"] < 0:
            exit_flags += 1
            profit_protection_triggered = True
            reasons.append("profit_protection_after_10pct_gain")
        if has_cost_basis and row["PnLPct"] > 15 and row["LastPrice"] < row["MA20"]:
            review_flags += 1
            reasons.append("gave_back_strength_after_15pct_gain")
        if has_cost_basis and row["PnLPct"] > 20 and row["Runup10dPct"] > 15:
            review_flags += 1
            reasons.append("large_gain_after_hot_runup")
        if has_cost_basis and structural_breakdown and row["PnLPct"] >= 0:
            review_flags += 1
            reasons.append("winner_in_pullback_but_not_broken")
        if row["LastPrice"] < row["MA20"] and row["MACDHist"] < 0:
            review_flags += 1
            reasons.append("losing_short_term_momentum")
        if (
            pd.notna(breakout_age)
            and breakout_age <= MAX_BREAKOUT_AGE
            and pd.notna(dist_from_breakout_pct)
            and dist_from_breakout_pct < 0
            and row["LastPrice"] < row["MA20"]
        ):
            review_flags += 1
            reasons.append("fresh_breakout_failed_to_hold_pivot")
        if (
            pd.notna(breakout_age)
            and breakout_age <= MAX_BREAKOUT_AGE
            and pd.notna(dist_from_breakout_pct)
            and dist_from_breakout_pct > MAX_BREAKOUT_EXTENSION * 100
            and has_cost_basis
            and row["PnLPct"] >= 10
        ):
            review_flags += 1
            reasons.append("fresh_breakout_extended_beyond_entry_window")
        if (
            scanner_signal_type == "SETUP"
            and row["LastPrice"] < row["MA20"]
            and row["MA20Slope"] < 0
        ):
            review_flags += 1
            reasons.append("setup_quality_deteriorating")
        if pd.notna(scanner_score) and scanner_score < review_threshold:
            review_flags += 1
            reasons.append("scanner_hold_quality_weak")

    else:
        if macro_regime == "Risk-on":
            review_flags += 1
            reasons.append("macro_risk_on_backdrop")
        if row["LastPrice"] > row["MA50"] and row["LastPrice"] > row["MA20"] and row["MA20Slope"] > 0:
            exit_flags += 1
            reasons.append("short_against_rising_trend")
        if row["MACDHist"] > 0 and row["LastPrice"] > row["MA20"]:
            exit_flags += 1
            reasons.append("positive_macd_above_ma20")
        if pd.notna(row["PnLPct"]) and row["PnLPct"] < -8:
            exit_flags += 1
            reasons.append("loss_exceeds_8pct")
        if pd.notna(row["PnLPct"]) and row["PnLPct"] >= 10 and row["LastPrice"] > row["MA20"] and row["MACDHist"] > 0:
            exit_flags += 1
            profit_protection_triggered = True
            reasons.append("profit_protection_after_10pct_gain")
        if row["LastPrice"] > row["MA20"] and row["MACDHist"] > 0:
            review_flags += 1
            reasons.append("short_losing_momentum_edge")

    unique_reasons = "; ".join(dict.fromkeys(reasons))
    if exit_flags >= 2:
        recommendation = "EXIT"
        reason = unique_reasons
    elif profit_protection_triggered:
        recommendation = "EXIT"
        reason = unique_reasons
    elif exit_flags == 1 or review_flags >= 1:
        recommendation = "REVIEW"
        reason = unique_reasons
    elif row["Side"] == "LONG":
        if pd.notna(row.get("ScannerScore", np.nan)) and row.get("ScannerScore", np.nan) < hold_threshold:
            recommendation = "REVIEW"
            reason = "scanner_hold_quality_marginal"
        elif row.get("ScannerSignalType") == "SETUP":
            recommendation = "HOLD"
            reason = "scanner_setup_still_valid"
        elif row.get("ScannerSignalType") == "BREAKOUT":
            recommendation = "HOLD"
            reason = "scanner_breakout_still_valid"
        else:
            recommendation = "HOLD"
            reason = "trend_and_momentum_still_support_position"
    else:
        recommendation = "HOLD"
        reason = "trend_and_momentum_still_support_position"

    exit_pressure = float(np.clip(0.32 * exit_flags + 0.12 * review_flags + (0.22 if profit_protection_triggered else 0.0), 0.0, 1.0))
    if recommendation == "HOLD":
        exit_pressure = min(exit_pressure, 0.35)
    elif recommendation == "REVIEW":
        exit_pressure = max(exit_pressure, 0.4)
        exit_pressure = min(exit_pressure, 0.79)
    else:
        exit_pressure = max(exit_pressure, 0.8)

    return {
        "recommendation": recommendation,
        "reason": reason,
        "exit_pressure": round(exit_pressure, 3),
        "exit_flags": exit_flags,
        "review_flags": review_flags,
        "profit_protection_triggered": profit_protection_triggered,
    }


def build_recommendation(
    row: pd.Series,
    leveraged_flag: bool,
    macro_snapshot: dict[str, object],
) -> tuple[str, str]:
    assessment = assess_exit_state(row, leveraged_flag, macro_snapshot)
    return str(assessment["recommendation"]), str(assessment["reason"])


def build_exit_pressure(
    row: pd.Series,
    leveraged_flag: bool,
    macro_snapshot: dict[str, object],
) -> float:
    assessment = assess_exit_state(row, leveraged_flag, macro_snapshot)
    return float(assessment["exit_pressure"])


def build_option_recommendation(row: pd.Series) -> tuple[str, str]:
    reasons: list[str] = []
    exit_flags = 0
    review_flags = 0

    option_is_call = row["InstrumentType"] == "CALL"
    pnl_pct = row["PnLPct"]
    underlying_above_ma20 = row["LastPrice"] > row["MA20"]
    underlying_above_ma50 = row["LastPrice"] > row["MA50"]
    macd_positive = row["MACDHist"] > 0

    if pd.notna(pnl_pct) and pnl_pct < -50:
        exit_flags += 1
        reasons.append("option_loss_exceeds_50pct")
    elif pd.notna(pnl_pct) and pnl_pct < -25:
        review_flags += 1
        reasons.append("option_loss_exceeds_25pct")

    if option_is_call:
        if not underlying_above_ma20 and not underlying_above_ma50:
            exit_flags += 1
            reasons.append("underlying_below_ma20_and_ma50")
        if not underlying_above_ma20 and not macd_positive:
            review_flags += 1
            reasons.append("underlying_lost_short_term_momentum")
    else:
        if underlying_above_ma20 and underlying_above_ma50:
            exit_flags += 1
            reasons.append("underlying_above_ma20_and_ma50")
        if underlying_above_ma20 and macd_positive:
            review_flags += 1
            reasons.append("underlying_lost_short_term_momentum")

    if pd.notna(row["Runup10dPct"]) and abs(row["Runup10dPct"]) > 15:
        review_flags += 1
        reasons.append("underlying_moved_sharply_recently")

    if exit_flags >= 2:
        return "EXIT", "; ".join(dict.fromkeys(reasons))
    if exit_flags == 1 or review_flags >= 1:
        return "REVIEW", "; ".join(dict.fromkeys(reasons))
    return "HOLD", "underlying_trend_still_supports_option"


def review_positions(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return positions

    macro_snapshot = get_macro_regime_snapshot()
    leveraged_flags = build_metadata_flags(positions["Symbol"].tolist())
    rows = []

    for row in positions.itertuples(index=False):
        news = get_symbol_news_sentiment(row.Symbol)
        hist = download_history(row.Symbol)
        if hist is None or hist.empty:
            rows.append(
                {
                    **row._asdict(),
                    "NewsSentiment": news["sentiment_score"],
                    "NewsHeadlineCount": news["headline_count"],
                    "NewsHeadlinePreview": " | ".join(news["headlines"]),
                    "Recommendation": "REVIEW",
                    "Reason": "no_market_data",
                }
            )
            continue

        last = hist.iloc[-1]
        underlying_last_price = float(last["Close"])
        if getattr(row, "InstrumentType", "EQUITY") in {"CALL", "PUT"}:
            option_mark = float(row.OptionMark) if pd.notna(row.OptionMark) else np.nan
            last_price = option_mark
            if pd.notna(row.CostBasis) and pd.notna(option_mark):
                pnl_pct = (option_mark / row.CostBasis - 1) * 100
            else:
                pnl_pct = np.nan
        else:
            last_price = underlying_last_price
            if pd.notna(row.CostBasis):
                pnl_pct = ((underlying_last_price / row.CostBasis) - 1) * 100 if row.Side == "LONG" else ((row.CostBasis / underlying_last_price) - 1) * 100
            else:
                pnl_pct = np.nan

        review_row = pd.Series(
            {
                "InstrumentType": getattr(row, "InstrumentType", "EQUITY"),
                "Side": row.Side,
                "LastPrice": underlying_last_price,
                "MA20": float(last["ma20"]) if pd.notna(last["ma20"]) else np.nan,
                "MA50": float(last["ma50"]) if pd.notna(last["ma50"]) else np.nan,
                "MA150": float(last["ma150"]) if pd.notna(last["ma150"]) else np.nan,
                "PnLPct": pnl_pct,
                "Runup10dPct": float(last["runup_10d_pct"]) if pd.notna(last["runup_10d_pct"]) else np.nan,
                "AvgDollarVol20dM": float(last["avg_dollar_vol_20d"]) / 1_000_000 if pd.notna(last["avg_dollar_vol_20d"]) else np.nan,
                "MACDHist": float(last["macd_hist"]) if pd.notna(last["macd_hist"]) else np.nan,
                "MA20Slope": float(last["ma20_slope"]) if pd.notna(last["ma20_slope"]) else np.nan,
                "BreakoutAge": float(last["breakout_age"]) if pd.notna(last["breakout_age"]) else np.nan,
                "DistFromMA20Pct": float(last["extension_from_ma20"] * 100) if pd.notna(last["extension_from_ma20"]) else np.nan,
                "DistFromBreakoutPct": float(last["extension_from_breakout"] * 100) if pd.notna(last["extension_from_breakout"]) else np.nan,
                "ScannerScore": float(last["scanner_score"]) if pd.notna(last["scanner_score"]) else np.nan,
                "ScannerTrend": bool(last["scanner_trend"]) if pd.notna(last["scanner_trend"]) else False,
                "ScannerSignalType": (
                    "SETUP"
                    if bool(last["setup_candidate"])
                    else "BREAKOUT" if bool(last["breakout_candidate"]) else ""
                ),
            }
        )

        if getattr(row, "InstrumentType", "EQUITY") in {"CALL", "PUT"}:
            recommendation, reason = build_option_recommendation(review_row)
        else:
            recommendation, reason = build_recommendation(
                review_row,
                bool(leveraged_flags.get(row.Symbol, False)),
                macro_snapshot,
            )

        rows.append(
            {
                "Symbol": row.Symbol,
                "InstrumentType": getattr(row, "InstrumentType", "EQUITY"),
                "Side": row.Side,
                "Quantity": round(float(row.Quantity), 4) if pd.notna(row.Quantity) else np.nan,
                "CostBasis": round(float(row.CostBasis), 4) if pd.notna(row.CostBasis) else np.nan,
                "LastPrice": round(last_price, 4),
                "UnderlyingLastPrice": round(underlying_last_price, 4),
                "Expiration": getattr(row, "Expiration", None),
                "Strike": round(float(getattr(row, "Strike", np.nan)), 4) if pd.notna(getattr(row, "Strike", np.nan)) else np.nan,
                "PnLPct": round(pnl_pct, 2) if pd.notna(pnl_pct) else np.nan,
                "MA20": round(float(last["ma20"]), 4) if pd.notna(last["ma20"]) else np.nan,
                "MA50": round(float(last["ma50"]), 4) if pd.notna(last["ma50"]) else np.nan,
                "MA150": round(float(last["ma150"]), 4) if pd.notna(last["ma150"]) else np.nan,
                "Runup10dPct": round(float(last["runup_10d_pct"]), 2) if pd.notna(last["runup_10d_pct"]) else np.nan,
                "AvgDollarVol20dM": round(float(last["avg_dollar_vol_20d"]) / 1_000_000, 2) if pd.notna(last["avg_dollar_vol_20d"]) else np.nan,
                "MACDHist": round(float(last["macd_hist"]), 4) if pd.notna(last["macd_hist"]) else np.nan,
                "ScannerScore": round(float(review_row["ScannerScore"]), 3) if pd.notna(review_row["ScannerScore"]) else np.nan,
                "ScannerSignalType": review_row["ScannerSignalType"] or pd.NA,
                "BreakoutAge": round(float(review_row["BreakoutAge"]), 1) if pd.notna(review_row["BreakoutAge"]) else np.nan,
                "DistFromMA20Pct": round(float(review_row["DistFromMA20Pct"]), 2) if pd.notna(review_row["DistFromMA20Pct"]) else np.nan,
                "DistFromBreakoutPct": round(float(review_row["DistFromBreakoutPct"]), 2) if pd.notna(review_row["DistFromBreakoutPct"]) else np.nan,
                "MacroRegime": str(macro_snapshot["regime"]),
                "MacroScore": int(macro_snapshot["score"]),
                "MacroTrend": str(macro_snapshot["trend"]),
                "MacroRisk": str(macro_snapshot["risk"]),
                "NewsSentiment": news["sentiment_score"],
                "NewsHeadlineCount": news["headline_count"],
                "NewsHeadlinePreview": " | ".join(news["headlines"]),
                "Recommendation": recommendation,
                "Reason": reason,
                "AsOfDate": last.name.date(),
                "EarningsDate": getattr(row, "EarningsDate", None),
                "FirstEntryDate": row.FirstEntryDate.date() if pd.notna(row.FirstEntryDate) else None,
            }
        )

    return pd.DataFrame(rows).sort_values(["Recommendation", "PnLPct", "Symbol"]).reset_index(drop=True)


def review_positions_from_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    lines = path.read_text().splitlines()

    if any(line.startswith("Activity/Trade Date,") for line in lines[:15]):
        txns = load_transactions(csv_path)
        positions = reconstruct_open_positions(txns)
    else:
        positions = load_simple_positions(csv_path)

    return review_positions(positions)


def main() -> None:
    args = parse_args()
    review_df = review_positions_from_csv(args.csv_path)
    review_df.to_csv(args.output, index=False)

    if review_df.empty:
        print("No open positions found.")
        return

    print("Current Position Review")
    print(review_df.to_string(index=False))
    print("")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
