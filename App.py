import time
import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np

@st.cache_data(show_spinner=False)
def fetch_company_meta(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info  # metadata call
            name = info.get("shortName") or info.get("longName") or ""
            sector = info.get("sector") or ""
            industry = info.get("industry") or ""
            rows.append({"Ticker": t, "Name": name, "Sector": sector, "Industry": industry})
        except Exception:
            rows.append({"Ticker": t, "Name": "", "Sector": "", "Industry": ""})
    return pd.DataFrame(rows)


CSV1 = "nasdaq_100_spy_500.csv"
CSV2 = "my_universe.csv"
TICKER_COL = "Symbol"

st.set_page_config(page_title="Weekly Shortlist", layout="wide")

@st.cache_data(show_spinner=False)
def load_all_tickers() -> list[str]:
    df1 = pd.read_csv(CSV1)
    df2 = pd.read_csv(CSV2)

    tickers = pd.concat([df1[TICKER_COL], df2[TICKER_COL]], ignore_index=True).dropna()
    tickers = tickers.astype(str).str.strip().str.upper()
    tickers = tickers[tickers != ""]
    return sorted(tickers.unique())

def compute_ma20_qualified(tickers: list[str], period: str = "3mo") -> pd.DataFrame:
    # Fetch prices (many tickers at once)
    px = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )

    rows = []
    multi = isinstance(px.columns, pd.MultiIndex)

    for t in tickers:
        try:
            close = px[(t, "Close")].dropna() if multi else px["Close"].dropna()
            if len(close) < 25:
                continue

            ma20 = close.rolling(20).mean()
            last_close = float(close.iloc[-1])
            last_ma20 = float(ma20.iloc[-1])
            ma20_last = ma20.dropna()
            if len(ma20_last) < 25:
                continue

            N = 5
            y = ma20_last.iloc[-N:].to_numpy()
            x = np.arange(N)
            slope = np.polyfit(x, y, 1)[0]  # linear regression slope
            not_extended = last_close <= last_ma20 * 1.02
            ma20_series = ma20.dropna()
            close_series = close.loc[ma20_series.index]  # align indices

            # Need at least 5+ days of MA20 to classify context
            K = 5
            if len(ma20_series) < (K + 1):
                continue

            close0 = float(close_series.iloc[-1])
            ma0 = float(ma20_series.iloc[-1])

            close_k = float(close_series.iloc[-(K + 1)])
            ma_k = float(ma20_series.iloc[-(K + 1)])

            dist_pct = (close0 / ma0 - 1.0) * 100.0  # % above MA20

            if (close0 > ma0) and (close_k <= ma_k):
                context = "Breakout"
            elif (close0 > ma0) and (close_k > ma_k) and (dist_pct <= 2.0):
                context = "Pullback"
            else:
                context = "Other"

            not_extended = close0 <= ma0 * 1.03

            qualified = (close0 > ma0) and (slope >= 0) and not_extended

            rows.append({
                "Ticker": t,
                "Close": close0,
                "MA20": ma0,
                "MA20_slope_5d": slope,
                "Dist_to_MA20_%": dist_pct,
                "Context": context,
                "Qualified": qualified,
            })


        except Exception:
            # Skip symbols that fail
            continue

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Qualified", "Ticker"], ascending=[False, True]).reset_index(drop=True)
    return out

def init_state():
    st.session_state.setdefault("mode", "Pullback")
    st.session_state.setdefault("event_note", "")
    st.session_state.setdefault("event_flag", False)
    st.session_state.setdefault("checklist", {
        "1. Choose weekly market mode (pullback vs breakout)": False,
        "2. Check major upcoming market events": False,
        "3. Remove excessive volatility (>10% avg move)": False,
        "4. Flag stocks extended far from MA20 (attention only)": False,
        "5. Exclude stocks below your market-cap minimum": False,
        "6. Group by industry to check concentration": False,
        "7. Only this universe until next weekly reset": False,
    })

init_state()

# ---------- Sidebar ----------
st.sidebar.header("Weekly Reset")

st.sidebar.selectbox("Mode", ["Pullback", "Breakout"], key="mode")
st.sidebar.checkbox("Any major upcoming event?", key="event_flag")
st.sidebar.text_area("Event notes", key="event_note", height=80)

st.sidebar.divider()
st.sidebar.subheader("Checklist")
for k in list(st.session_state["checklist"].keys()):
    st.session_state["checklist"][k] = st.sidebar.checkbox(k, value=st.session_state["checklist"][k])

st.sidebar.divider()

all_tickers = load_all_tickers()
st.sidebar.caption(f"Universe size (merged): {len(all_tickers)} tickers")

if st.sidebar.button("Reset for New Week (Compute MA20)", type="primary"):
    with st.spinner("Downloading daily closes and computing MA20..."):
        t0 = time.time()
        df = compute_ma20_qualified(all_tickers, period="3mo")
        st.session_state["universe_df"] = df
        st.session_state["reset_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["elapsed"] = round(time.time() - t0, 2)

# ---------- Main ----------
st.title("This Week’s Qualified Shortlist")

if "universe_df" not in st.session_state:
    st.info("Click **Reset for New Week** in the sidebar to compute the MA20-qualified shortlist.")
else:
    df = st.session_state["universe_df"]
    qualified = df[df["Qualified"]].copy()
    meta = fetch_company_meta(qualified["Ticker"].tolist())
    qualified = qualified.merge(meta, on="Ticker", how="left")

    st.caption(
        f"Last reset: {st.session_state.get('reset_time')} "
        f"(computed in {st.session_state.get('elapsed')}s) • "
        f"Mode: {st.session_state['mode']} • "
        f"Qualified: {len(qualified)} / {len(df)}"
    )

    cols = [
        "Ticker", "Name", "Sector", "Industry",
        "Context",
        "Dist_to_MA20_%", "MA20_slope_5d",
        "Close", "MA20",
    ]
    st.dataframe(qualified[cols], use_container_width=True, hide_index=True)

    with st.expander("Show full universe (debug)"):
        st.dataframe(df, use_container_width=True, hide_index=True)
