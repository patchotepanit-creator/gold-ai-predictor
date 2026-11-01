import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Gold AI Predictor", page_icon="🥇", layout="wide")

st.title("🥇 AI Gold Trading Price Predictor")
st.markdown("""
**Educational tool only.** Predictions are based on historical patterns and are **not financial advice**. 
Gold markets are influenced by geopolitics, interest rates, inflation, USD strength, and many other factors. 
Past performance does not guarantee future results. Always do your own research.
""")

# Sidebar
st.sidebar.header("Settings")
lookback_days = st.sidebar.slider("Historical days to use", 200, 1200, 600)
forecast_days = st.sidebar.slider("Forecast horizon (days)", 1, 7, 1)
n_estimators = st.sidebar.slider("Random Forest trees", 50, 300, 100)
retrain = st.sidebar.button("🔄 Retrain Model & Refresh")

@st.cache_data(ttl=3600)
def fetch_gold_history():
    try:
        r = requests.get("https://xaus.com/api/v1/history", timeout=15)
        r.raise_for_status()
        data = r.json()
        points = data.get("points", [])
        df = pd.DataFrame(points)
        df["date"] = pd.to_datetime(df["d"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.rename(columns={"c": "close", "h": "high", "l": "low"})
        df = df[["date", "close", "high", "low"]].dropna()
        return df
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_live_price():
    try:
        r = requests.get("https://xaus.com/api/v1/spot", timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("spot_usd_oz") or data.get("xau", {}).get("price")
    except:
        return None

def add_features(df):
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["volatility"] = df["return"].rolling(10).std()
    # Simple RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["high_low_spread"] = (df["high"] - df["low"]) / df["close"]
    # Lags
    for lag in [1, 2, 3, 5]:
        df[f"close_lag{lag}"] = df["close"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

def train_and_predict(df, horizon=1):
    feature_cols = ["ma5", "ma10", "ma20", "volatility", "rsi", "high_low_spread",
                    "close_lag1", "close_lag2", "close_lag3", "close_lag5", "return"]
    target = "close"
    
    # Create target shifted by horizon
    df = df.copy()
    df["target"] = df["close"].shift(-horizon)
    df = df.dropna().reset_index(drop=True)
    
    X = df[feature_cols]
    y = df["target"]
    
    # Simple train/test split (last 20% for evaluation)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluation
    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    
    # Latest prediction
    latest_features = X.iloc[[-1]]
    next_pred = model.predict(latest_features)[0]
    
    return model, next_pred, mae, r2, df

# Main
df_raw = fetch_gold_history()
if df_raw is None or len(df_raw) < 50:
    st.error("Could not load gold price history. Please try again later.")
    st.stop()

df_raw = df_raw.tail(lookback_days).reset_index(drop=True)
df = add_features(df_raw)

if len(df) < 30:
    st.warning("Not enough data after feature engineering.")
    st.stop()

if retrain or "model_result" not in st.session_state:
    with st.spinner("Training AI model..."):
        model, next_pred, mae, r2, df_feat = train_and_predict(df, forecast_days)
        st.session_state.model_result = (model, next_pred, mae, r2, df_feat)
else:
    model, next_pred, mae, r2, df_feat = st.session_state.model_result

live_price = fetch_live_price()
current_close = df_raw["close"].iloc[-1]
display_price = live_price if live_price else current_close

# Layout
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Gold Price", f"${display_price:,.2f}")
col2.metric(f"AI Predicted ({forecast_days}d ahead)", f"${next_pred:,.2f}")
delta = next_pred - display_price
delta_pct = (delta / display_price) * 100
col3.metric("Expected Change", f"${delta:+.2f}", f"{delta_pct:+.2f}%")
signal = "🟢 BULLISH" if delta > 0 else "🔴 BEARISH"
col4.metric("Signal", signal)

st.markdown(f"**Model quality (on recent hold-out):** MAE ≈ ${mae:.1f} | R² ≈ {r2:.3f}")

# Chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_raw["date"], y=df_raw["close"],
    mode="lines", name="Historical Close",
    line=dict(color="#FFD700", width=2)
))
# Add prediction point
last_date = df_raw["date"].iloc[-1]
future_date = last_date + pd.Timedelta(days=forecast_days)
fig.add_trace(go.Scatter(
    x=[last_date, future_date],
    y=[display_price, next_pred],
    mode="lines+markers", name="AI Forecast",
    line=dict(color="#00FF88", width=3, dash="dash"),
    marker=dict(size=10)
))
fig.update_layout(
    title="Gold Price History + AI Forecast",
    xaxis_title="Date",
    yaxis_title="Price (USD / troy oz)",
    template="plotly_dark",
    height=500,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# Recent data table
with st.expander("Recent price data"):
    st.dataframe(df_raw.tail(15).style.format({"close": "${:.2f}", "high": "${:.2f}", "low": "${:.2f}"}))

st.markdown("---")
st.caption("Data source: xaus.com free API • Model: Random Forest Regressor on technical features • Built with Streamlit")
st.caption("⚠️ This is a simple prototype. Real trading systems use more data, better features, walk-forward validation, risk management, and human oversight.")
