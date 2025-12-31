import time
import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import joblib

@st.cache_resource
def load_ml_models():
    model_up = joblib.load("models/model_up.pkl")
    model_down = joblib.load("models/model_down.pkl")
    return model_up, model_down

@st.cache_data(show_spinner=False)
def fetch_one_meta(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "Name": info.get("shortName") or info.get("longName") or "",
            "Sector": info.get("sector") or "",
            "Industry": info.get("industry") or "",
        }
    except Exception:
        return {"Name": "", "Sector": "", "Industry": ""}

@st.cache_data(show_spinner=False)
def fetch_daily_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    return df.dropna()

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

def build_ml_features(pack: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "Dist_to_MA20_%": pack.get("Dist_to_MA20_%"),
        "MA20_slope_5d": pack.get("MA20_slope_5d"),
        "UDR10": pack.get("UDR10"),
        "TPS10": pack.get("TPS10"),
        "VolRatio_5_20": pack.get("VolRatio_5_20"),
        "Context": pack.get("Context"),
    }])

def trend_persistence_score(hist: pd.DataFrame, n: int = 10) -> float:
    if hist is None or hist.empty:
        return float("nan")
    if ("Close" not in hist.columns) or ("Low" not in hist.columns):
        return float("nan")

    close = hist["Close"]
    low = hist["Low"]

    # If yfinance returns multiple columns (DataFrame), take the first
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]

    close = close.astype(float)
    low = low.astype(float)

    ma20 = close.rolling(20).mean()

    valid = pd.concat([low, ma20], axis=1).dropna()
    valid.columns = ["Low", "MA20"]

    if len(valid) < n:
        return float("nan")

    recent = valid.tail(n)
    return float((recent["Low"] >= recent["MA20"]).mean())

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
#            slope_pct = (y / last_ma20) * 100 if last_ma20 else float("nan")

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
#                "MA20_slope_pct": slope_pct,

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

def up_day_ratio(close: pd.Series, n: int = 10) -> float:
    """
    % of up days in last n sessions, where Close[t] > Close[t-1]
    Returns a fraction in [0,1]. Needs n+1 closes.
    """
    if len(close) < n + 1:
        return float("nan")
    last = close.iloc[-(n+1):]
    ups = (last.diff() > 0).sum()
    return float(ups) / n

def compute_trend_pack(hist: pd.DataFrame) -> dict:
    close = hist["Close"]
    udr10 = up_day_ratio(close, n=10)
    last = float(close.iloc[-1])

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    last_ma20 = float(ma20.iloc[-1])
    print(last_ma20)
    last_ma50 = float(ma50.iloc[-1]) #if not pd.isna(ma50.iloc[-1]) else float("nan")
    last_ma200 = float(ma200.iloc[-1]) #if not pd.isna(ma200.iloc[-1]) else float("nan")

    dist20 = (last / last_ma20 - 1.0) * 100.0

    # returns
    r1w = (last / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 else float("nan")
    r1m = (last / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else float("nan")

    # badge (simple, explainable)
    if (last > last_ma20) and (not pd.isna(last_ma50)) and (last_ma20 > last_ma50):
        badge = "Bull"
    elif (last < last_ma20) and (not pd.isna(last_ma50)) and (last_ma20 < last_ma50):
        badge = "Weak"
    else:
        badge = "Mixed"

    tps10 = trend_persistence_score(hist, 10)
    N = 5
    y = ma20.iloc[-N:].to_numpy()
    x = np.arange(N)
    slope = np.polyfit(x, y, 1)[0]  # linear regression slope
    print(slope)
#    slope_pct = (y / last_ma20) * 100 if last_ma20 else float("nan")

    return {
        "badge": badge,
        "Close": last,
        "MA20": last_ma20,
        "MA20_slope_5d": slope[0],
 #       "MA20_slope_pct": slope_pct,

        "MA50": last_ma50,
        "MA200": last_ma200,
        "Dist_to_MA20_%": dist20,
        "Return_1W_%": r1w,
        "Return_1M_%": r1m,
        "UDR10": udr10,
        "TPS10": tps10,

    }

def confidence_flag(pack: dict) -> tuple[str, int, list[str]]:
    """
    Returns (label, score, reasons)
    label in {"🟢 Favorable", "🟡 Mixed", "🔴 Caution"}
    """
    score = 0
    reasons = []

    # Gate 1: above MA20
    if pack["Close"] > pack["MA20"]:
        score += 1
        reasons.append("+ Close above MA20")
    else:
        reasons.append("- Close below MA20")

    # Gate 2: MA20 slope (5d) non-negative
    slope=(pack.get("MA20_slope_5d", float("nan")))
    if slope >= 0:
        score += 1
        reasons.append(f"+ MA20 slope ≥ 0 ({slope:.2f}/ day)")
    else:
        slope_pct = pack.get("MA20_slope_pct", float("nan"))

        reasons.append(f"+ MA20 slope <= 0 ({slope_pct:.2f}% / day)")

    # Gate 3: UDR10 bias
    print(pack.get("UDR10", float("nan")))
    if pack.get("UDR10", float("nan")) >= 0.55:
        score += 1
        reasons.append("+ UDR10 ≥ 55%")
    else:
        reasons.append("- UDR10 < 55%")

    # Gate 4: TPS10 integrity
    if pack.get("TPS10", float("nan")) >= 0.70:
        score += 1
        reasons.append("+ TPS10 ≥ 70%")
    else:
        reasons.append("- TPS10 < 70%")

    # Penalty: extended (even if you cap in shortlist, typed deep dive may violate)
    dist = pack.get("Dist_to_MA20_%", float("nan"))
    if dist == dist and dist > 2.0:   # NaN-safe check: dist==dist
        score -= 1
        reasons.append("− Extended > 2% above MA20")

    if score >= 3:
        label = "🟢 Favorable"
    elif score >= 1:
        label = "🟡 Mixed"
    else:
        label = "🔴 Caution"

    return label, score, reasons

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

st.subheader("Selected Stock Deep Dive")

st.subheader("Selected Stock Deep Dive")

scope = st.radio(
    "Deep dive scope",
    ["Shortlist only", "Any ticker (manual)"],
    horizontal=True,
)

if scope == "Shortlist only":
    selected = st.selectbox(
        "Pick a ticker from this week's shortlist",
        qualified["Ticker"].tolist(),
    )
else:
    selected = st.text_input(
        "Type a ticker to deep dive (e.g., AAPL)",
        value="",
    ).strip().upper()

    if selected and (selected not in all_tickers):
        st.warning("Ticker not in your universe CSVs, but I’ll still try to fetch it.")
    meta1 = fetch_one_meta(selected)
    if meta1["Name"]:
        st.write(f"**{meta1['Name']}** — {meta1['Sector']} / {meta1['Industry']}")
    else:
        st.write(f"**{selected}**")

hist = fetch_daily_history(selected, period="3mo")


pack = compute_trend_pack(hist)
flag, score, reasons = confidence_flag(pack)
st.markdown(f"## {flag}  (score: {score})")

with st.expander("Why this flag?"):
    for r in reasons:
        st.write(r)

udr = pack["UDR10"]
if np.isnan(udr):
    bias = "Insufficient data"
elif udr >= 0.55:
    bias = "Upward bias"
elif udr <= 0.45:
    bias = "Fragile / downward pressure"
else:
    bias = "Neutral"

st.write(f"**UDR-10 bias:** {bias} ({udr*100:.0f}% up days)")

st.write(f"**Trend State:** {pack['badge']}")

tps = pack["TPS10"]
if np.isnan(tps):
    tps_label = "Insufficient data"
elif tps >= 0.70:
    tps_label = "Strong MA20 integrity"
elif tps >= 0.50:
    tps_label = "Normal"
else:
    tps_label = "Weak / sloppy"

st.write(f"**TPS-10:** {tps_label} ({tps*100:.0f}% of days Low stayed above MA20)")

c1, c2, c3 = st.columns(3)
c1.metric("Close", f"{pack['Close']:.2f}")
c2.metric("MA20", f"{pack['MA20']:.2f}", f"{pack['Dist_to_MA20_%']:.2f}% vs MA20")
c3.metric("1W / 1M", f"{pack['Return_1W_%']:.2f}%", f"{pack['Return_1M_%']:.2f}%")

# optional: show MA50/MA200
st.caption(f"MA50: {pack['MA50']:.2f} • MA200: {pack['MA200']:.2f}")

model_up, model_down = load_ml_models()
X_ml = build_ml_features(pack)

p_up = model_up.predict_proba(X_ml)[0, 1]
p_down = model_down.predict_proba(X_ml)[0, 1]
BASE_UP = 0.1266
BASE_DOWN = 0.1177

def label(p, base):
    if p >= base + 0.05:
        return "above average"
    elif p <= base - 0.05:
        return "below average"
    else:
        return "normal"

st.subheader("📊 Next-week large-move risk (research)")

st.write(
    f"↑ **Upside ≥ +3%**: {p_up:.1%} ({label(p_up, BASE_UP)})  \n"
    f"↓ **Downside ≤ −3%**: {p_down:.1%} ({label(p_down, BASE_DOWN)})"
)

st.caption("Probabilities are pooled, weekly-horizon risk estimates. Research only.")

