import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="Gold AI Trader", page_icon=":bar_chart:", layout="wide")

st.title("AI Gold Day-to-Day Trader")
st.markdown("""
**Educational tool only — not financial advice.**  
Uses technical analysis + macro factors (DXY, Silver, VIX, Oil) to analyze gold and forecast short-term moves.
""")

# Sidebar
st.sidebar.header("Settings")
lookback = st.sidebar.slider("Lookback days", 200, 1000, 400)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 1, 30, 5)
retrain = st.sidebar.button("Retrain & Refresh")

@st.cache_data(ttl=3600)
def fetch_gold_history():
    try:
        r = requests.get("https://xaus.com/api/v1/history", timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data.get("points", []))
        df["date"] = pd.to_datetime(df["d"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.rename(columns={"c": "close", "h": "high", "l": "low"})
        return df[["date", "close", "high", "low"]].dropna()
    except Exception as e:
        st.error(f"Gold data error: {e}")
        return None

@st.cache_data(ttl=1800)
def fetch_macro_data(days=500):
    tickers = {
        "dxy": "DX-Y.NYB",
        "silver": "SI=F",
        "vix": "^VIX",
        "oil": "CL=F"
    }
    end = datetime.now()
    start = end - timedelta(days=days)
    macros = {}
    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not data.empty:
                macros[name] = data["Close"].rename(name)
        except:
            pass
    if macros:
        df = pd.concat(macros.values(), axis=1)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_live_price():
    try:
        r = requests.get("https://xaus.com/api/v1/spot", timeout=10)
        return r.json().get("spot_usd_oz") or r.json().get("xau", {}).get("price")
    except:
        return None

def add_technical_features(df):
    df = df.copy()
    df["return"] = df["close"].pct_change()
    for w in [5, 10, 20, 50]:
        df[f"ma{w}"] = df["close"].rolling(w).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["volatility"] = df["return"].rolling(10).std()
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    for lag in [1, 2, 3, 5]:
        df[f"close_lag{lag}"] = df["close"].shift(lag)
    df["resistance"] = df["high"].rolling(20).max()
    df["support"] = df["low"].rolling(20).min()
    return df.dropna().reset_index(drop=True)

def merge_macros(gold_df, macro_df):
    if macro_df.empty:
        return gold_df
    gold_df = gold_df.set_index("date")
    merged = gold_df.join(macro_df, how="left")
    for col in macro_df.columns:
        merged[col] = merged[col].ffill()
    for col in macro_df.columns:
        merged[f"{col}_ret"] = merged[col].pct_change()
    return merged.reset_index()

def train_predict(df, horizon=5):
    feature_cols = [
        "ma5", "ma10", "ma20", "ma50", "rsi", "macd", "macd_hist",
        "bb_width", "volatility", "hl_spread", "close_lag1", "close_lag2",
        "resistance", "support"
    ]
    macro_cols = [c for c in df.columns if c.endswith("_ret") or c in ["dxy", "silver", "vix", "oil"]]
    feature_cols += [c for c in macro_cols if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    df = df.copy()
    df["target"] = df["close"].shift(-horizon)
    df = df.dropna().reset_index(drop=True)

    if len(df) < 50:
        return None, None, None, None, None

    X = df[feature_cols]
    y = df["target"]
    split = int(len(df) * 0.8)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X.iloc[:split], y.iloc[:split])

    pred_test = model.predict(X.iloc[split:])
    mae = mean_absolute_error(y.iloc[split:], pred_test)
    r2 = r2_score(y.iloc[split:], pred_test)

    next_pred = model.predict(X.iloc[[-1]])[0]
    return model, next_pred, mae, r2, feature_cols

# Load data
gold_raw = fetch_gold_history()
if gold_raw is None or len(gold_raw) < 50:
    st.error("Could not load gold data.")
    st.stop()

gold_raw = gold_raw.tail(lookback).reset_index(drop=True)
macro_df = fetch_macro_data()
df = merge_macros(gold_raw, macro_df)
df = add_technical_features(df)

if retrain or "result" not in st.session_state:
    with st.spinner("Training model with technical + macro features..."):
        model, next_pred, mae, r2, feats = train_predict(df, forecast_horizon)
        st.session_state.result = (next_pred, mae, r2)
else:
    next_pred, mae, r2 = st.session_state.result

live = fetch_live_price()
current = gold_raw["close"].iloc[-1]
price = live if live else current
delta = next_pred - price if next_pred else 0
delta_pct = (delta / price * 100) if price else 0

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Gold", f"${price:,.2f}")
col2.metric(f"{forecast_horizon}-Day Forecast", f"${next_pred:,.2f}" if next_pred else "N/A")
col3.metric("Expected Move", f"${delta:+.2f}", f"{delta_pct:+.2f}%")
signal = "BULLISH" if delta > 0 else "BEARISH"
col4.metric("Signal", signal)

if mae:
    st.caption(f"Hold-out MAE: ${mae:.1f} | R2: {r2:.3f}")

# Charts
tab1, tab2, tab3 = st.tabs(["Price + Forecast", "Technical Analysis", "Macro Factors"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gold_raw["date"], y=gold_raw["close"], name="Gold Close", line=dict(color="#FFD700")))
    last_date = gold_raw["date"].iloc[-1]
    future = last_date + pd.Timedelta(days=forecast_horizon)
    fig.add_trace(go.Scatter(x=[last_date, future], y=[price, next_pred], name="AI Forecast",
                             line=dict(color="#00FF88", dash="dash", width=3), mode="lines+markers"))
    fig.update_layout(title="Gold Price + AI Forecast", template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                         row_heights=[0.5, 0.25, 0.25])
    fig2.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close", line=dict(color="#FFD700")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"], name="BB Upper", line=dict(color="gray", dash="dot")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"], name="BB Lower", line=dict(color="gray", dash="dot")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df["date"], y=df["support"], name="Support", line=dict(color="green", dash="dash")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df["date"], y=df["resistance"], name="Resistance", line=dict(color="red", dash="dash")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df["date"], y=df["macd"], name="MACD", line=dict(color="cyan")), row=2, col=1)
    fig2.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal", line=dict(color="orange")), row=2, col=1)
    fig2.add_trace(go.Scatter(x=df["date"], y=df["rsi"], name="RSI", line=dict(color="purple")), row=3, col=1)
    fig2.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig2.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig2.update_layout(height=700, template="plotly_dark", title="Technical Analysis")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    if not macro_df.empty:
        fig3 = go.Figure()
        for col in macro_df.columns:
            norm = macro_df[col] / macro_df[col].iloc[0] * 100
            fig3.add_trace(go.Scatter(x=macro_df.index, y=norm, name=col.upper()))
        fig3.update_layout(title="Macro Factors (Normalized to 100)", template="plotly_dark", height=450)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(macro_df.tail(10).style.format("{:.2f}"))
    else:
        st.warning("Macro data unavailable at the moment.")

st.markdown("---")
st.caption("Data: xaus.com (gold) + Yahoo Finance (DXY, Silver, VIX, Oil) • Model: Random Forest • Educational use only")
