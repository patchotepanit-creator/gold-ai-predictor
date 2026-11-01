import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from streamlit_autorefresh import st_autorefresh
import feedparser
from datetime import datetime, timedelta

st.set_page_config(page_title="Gold AI Trader", page_icon=":bar_chart:", layout="wide")

st.title("AI Gold Day-to-Day Trader")
st.markdown("""
**Educational tool only — not financial advice.**  
Uses technical analysis + expanded macro factors. Includes Kitco news headlines.
""")

# Sidebar
st.sidebar.header("Mode")
mode = st.sidebar.radio("Select Mode", ["Daily Forecast", "Day Trading (Intraday)"])

if mode == "Daily Forecast":
    lookback = st.sidebar.slider("Lookback days", 200, 1000, 400)
    forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 1, 30, 5)
else:
    interval = st.sidebar.selectbox("Timeframe", ["5m", "15m", "1h"], index=1)
    lookback_bars = st.sidebar.slider("Bars to load", 100, 500, 200)

st.sidebar.header("Auto Refresh")
auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=False)
refresh_sec = st.sidebar.slider("Refresh every (seconds)", 10, 120, 30)

if auto_refresh:
    st_autorefresh(interval=refresh_sec * 1000, key="gold_refresh")

st.sidebar.header("Alpaca Paper Trading")
alpaca_key = st.sidebar.text_input("API Key", type="password")
alpaca_secret = st.sidebar.text_input("API Secret", type="password")
paper_qty = st.sidebar.number_input("Shares (GLD)", min_value=1, max_value=100, value=1)
test_conn = st.sidebar.button("Test Alpaca Connection")
place_order = st.sidebar.button("Place Paper Order (GLD)")
force_buy = st.sidebar.button("Force Test BUY (1 GLD)")
force_sell = st.sidebar.button("Force Test SELL (1 GLD)")
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
    tickers = {"dxy": "DX-Y.NYB", "silver": "SI=F", "vix": "^VIX", "oil": "CL=F", "tnx": "^TNX", "fvx": "^FVX", "spy": "SPY", "copper": "HG=F", "btc": "BTC-USD", "eurusd": "EURUSD=X", "tip": "TIP"}
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
def fetch_intraday(interval="15m", bars=200):
    try:
        period = "5d" if interval in ["5m", "15m"] else "10d"
        data = yf.download("GC=F", period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty:
            return None
        df = data[["Open", "High", "Low", "Close"]].copy()
        df.columns = ["open", "high", "low", "close"]
        df = df.dropna().reset_index()
        df = df.rename(columns={"Datetime": "date", "Date": "date"})
        if "date" not in df.columns:
            df["date"] = df.index
        return df.tail(bars).reset_index(drop=True)
    except Exception as e:
        st.error(f"Intraday data error: {e}")
        return None

def place_alpaca_paper_order(api_key, api_secret, side, qty=1, symbol="GLD"):
    try:
        client = TradingClient(api_key, api_secret, paper=True)
        order_data = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY if side == "BUY" else OrderSide.SELL, time_in_force=TimeInForce.DAY)
        order = client.submit_order(order_data)
        return f"Paper order placed: {order.side} {order.qty} {order.symbol} (ID: {order.id})"
    except Exception as e:
        return f"Order failed: {str(e)}"

def get_gold_news(limit=8):
    try:
        feed = feedparser.parse("https://www.kitco.com/news/rss/kitconewsfeedextended.xml")
        news = []
        for entry in feed.entries[:limit]:
            news.append({"title": entry.get("title", ""), "link": entry.get("link", ""), "published": entry.get("published", "")})
        return news
    except:
        return []

def simple_news_bias(news_list):
    if not news_list:
        return "Neutral", 0
    bullish_words = ["rise", "higher", "rally", "gain", "surge", "bull", "support", "buy", "strong"]
    bearish_words = ["fall", "lower", "drop", "decline", "sell", "bear", "pressure", "weak", "loss"]
    score = 0
    for item in news_list:
        t = item["title"].lower()
        for w in bullish_words:
            if w in t:
                score += 1
        for w in bearish_words:
            if w in t:
                score -= 1
    if score > 1:
        return "Bullish bias from headlines", score
    elif score < -1:
        return "Bearish bias from headlines", score
    return "Neutral headlines", score

def get_alpaca_account(api_key, api_secret):
    try:
        client = TradingClient(api_key, api_secret, paper=True)
        acc = client.get_account()
        return f"Paper Buying Power: ${float(acc.buying_power):,.2f} | Equity: ${float(acc.equity):,.2f}"
    except Exception as e:
        return f"Account error: {str(e)}"

def generate_intraday_signal(df):
    if len(df) < 30:
        return None
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest["close"]
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    macd_cross_up = latest["macd"] > latest["macd_signal"] and prev["macd"] <= prev["macd_signal"]
    macd_cross_down = latest["macd"] < latest["macd_signal"] and prev["macd"] >= prev["macd_signal"]
    rsi_oversold = latest["rsi"] < 35
    rsi_overbought = latest["rsi"] > 65
    above_ma = price > latest["ma20"]
    if macd_cross_up or (rsi_oversold and above_ma):
        signal = "BUY"
        stop = price - 1.5 * atr
        target = price + 2.5 * atr
    elif macd_cross_down or (rsi_overbought and not above_ma):
        signal = "SELL"
        stop = price + 1.5 * atr
        target = price - 2.5 * atr
    else:
        signal = "HOLD"
        stop = price - atr
        target = price + atr
    return {"signal": signal, "price": price, "stop": stop, "target": target, "atr": atr, "rsi": latest["rsi"], "macd": latest["macd"]}

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
    if "silver" in merged.columns:
        merged["gold_silver_ratio"] = merged["close"] / merged["silver"]
    if "tnx" in merged.columns:
        merged["real_rate_proxy"] = merged["tnx"]
    return merged.reset_index()

def train_predict_multi(df, max_horizon=7):
    feature_cols = ["ma5", "ma10", "ma20", "ma50", "rsi", "macd", "macd_hist", "bb_width", "volatility", "hl_spread", "close_lag1", "close_lag2", "resistance", "support", "gold_silver_ratio"]
    macro_base = ["dxy", "silver", "vix", "oil", "tnx", "fvx", "spy", "copper", "btc", "eurusd", "tip"]
    macro_cols = [c for c in df.columns if c.endswith("_ret") or c in macro_base or c == "gold_silver_ratio"]
    feature_cols += [c for c in macro_cols if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]
    predictions = {}
    mae, r2 = None, None
    for h in range(1, max_horizon + 1):
        temp = df.copy()
        temp["target"] = temp["close"].shift(-h)
        temp = temp.dropna().reset_index(drop=True)
        if len(temp) < 50:
            continue
        X = temp[feature_cols]
        y = temp["target"]
        split = int(len(temp) * 0.8)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X.iloc[:split], y.iloc[:split])
        pred = model.predict(X.iloc[[-1]])[0]
        predictions[h] = pred
        if h == 1:
            pred_test = model.predict(X.iloc[split:])
            mae = mean_absolute_error(y.iloc[split:], pred_test)
            r2 = r2_score(y.iloc[split:], pred_test)
    return predictions, mae, r2

# ===== MAIN LOGIC =====
if mode == "Daily Forecast":
    gold_raw = fetch_gold_history()
    if gold_raw is None or len(gold_raw) < 50:
        st.error("Could not load gold data.")
        st.stop()
    gold_raw = gold_raw.tail(lookback).reset_index(drop=True)
    macro_df = fetch_macro_data()
    df = merge_macros(gold_raw, macro_df)
    df = add_technical_features(df)
    if retrain or "result" not in st.session_state:
        with st.spinner("Training day-by-day models..."):
            preds, mae, r2 = train_predict_multi(df, forecast_horizon)
            st.session_state.result = (preds, mae, r2)
    else:
        preds, mae, r2 = st.session_state.result
    live = None
    try:
        r = requests.get("https://xaus.com/api/v1/spot", timeout=10)
        live = r.json().get("spot_usd_oz") or r.json().get("xau", {}).get("price")
    except:
        pass
    current = gold_raw["close"].iloc[-1]
    price = live if live else current
    final_pred = preds.get(forecast_horizon, price)
    delta = final_pred - price
    delta_pct = (delta / price * 100) if price else 0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Gold", f"${price:,.2f}")
    col2.metric(f"{forecast_horizon}-Day Forecast", f"${final_pred:,.2f}")
    col3.metric("Expected Move", f"${delta:+.2f}", f"{delta_pct:+.2f}%")
    signal = "BULLISH" if delta > 0 else "BEARISH"
    col4.metric("Signal", signal)
    if mae:
        st.caption(f"1-day hold-out MAE: ${mae:.1f} | R2: {r2:.3f}")
    tab1, tab2, tab3 = st.tabs(["Price + Forecast", "Technical Analysis", "Macro Factors"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gold_raw["date"], y=gold_raw["close"], name="Gold Close", line=dict(color="#FFD700")))
        last_date = gold_raw["date"].iloc[-1]
        future_dates = [last_date] + [last_date + pd.Timedelta(days=h) for h in sorted(preds.keys())]
        future_prices = [price] + [preds[h] for h in sorted(preds.keys())]
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name="AI Day-by-Day Forecast", line=dict(color="#00FF88", dash="dash", width=3), mode="lines+markers"))
        fig.update_layout(title="Gold Price + Multi-Day AI Forecast", template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)
        if preds:
            table_data = []
            for h in sorted(preds.keys()):
                table_data.append({"Day": h, "Date": (last_date + pd.Timedelta(days=h)).strftime("%Y-%m-%d"), "Predicted Price": f"${preds[h]:,.2f}", "Change vs Today": f"{(preds[h] - price):+.2f} ({(preds[h]/price - 1)*100:+.2f}%)"})
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    with tab2:
        fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
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
else:
    st.subheader(f"Day Trading Mode — {interval} candles (Gold Futures GC=F)")
    intra = fetch_intraday(interval, lookback_bars)
    if intra is None or len(intra) < 30:
        st.error("Could not load intraday data. Try a different timeframe.")
        st.stop()
    intra = add_technical_features(intra)
    sig = generate_intraday_signal(intra)
    if sig:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${sig['price']:,.2f}")
        c2.metric("Signal", sig["signal"])
        c3.metric("Suggested Stop", f"${sig['stop']:,.2f}")
        c4.metric("Suggested Target", f"${sig['target']:,.2f}")
        st.caption(f"RSI: {sig['rsi']:.1f} | ATR: ${sig['atr']:.2f} | MACD: {sig['macd']:.2f}")
    st.subheader("Alpaca Paper Trading")
    if not alpaca_key or not alpaca_secret:
        st.warning("Enter API Key and Secret in the sidebar first.")
    else:
        status = get_alpaca_account(alpaca_key, alpaca_secret)
        if "error" in status.lower():
            st.error(status)
        else:
            st.success(status)
        if place_order:
            if sig and sig["signal"] in ["BUY", "SELL"]:
                result = place_alpaca_paper_order(alpaca_key, alpaca_secret, sig["signal"], int(paper_qty))
                if "placed" in result.lower():
                    st.success(result)
                else:
                    st.error(result)
            else:
                st.warning(f"No order sent — signal is {sig['signal'] if sig else 'none'}. Only BUY or SELL will place an order.")
        elif force_buy:
            result = place_alpaca_paper_order(alpaca_key, alpaca_secret, "BUY", 1)
            if "placed" in result.lower():
                st.success("FORCE TEST: " + result)
            else:
                st.error(result)
        elif force_sell:
            result = place_alpaca_paper_order(alpaca_key, alpaca_secret, "SELL", 1)
            if "placed" in result.lower():
                st.success("FORCE TEST: " + result)
            else:
                st.error(result)
        else:
            st.caption("Use 'Place Paper Order' (follows signal) or the Force Test buttons below.")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=intra["date"], open=intra["open"], high=intra["high"], low=intra["low"], close=intra["close"], name="Gold"))
    fig.add_trace(go.Scatter(x=intra["date"], y=intra["ma20"], name="MA20", line=dict(color="orange")))
    fig.update_layout(title=f"Gold Intraday ({interval})", template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Signals are rule-based (MACD cross + RSI). Stop/Target based on 1.5x / 2.5x ATR. Paper orders use GLD. Educational only.")

st.markdown("---")
st.subheader("Latest Gold News (Kitco)")
news = get_gold_news()
bias, score = simple_news_bias(news)
st.caption(f"Headline sentiment: {bias} (score {score})")

if news:
    for item in news:
        st.markdown(f"- [{item['title']}]({item['link']})  \n<small>{item['published']}</small>", unsafe_allow_html=True)
else:
    st.info("News feed temporarily unavailable.")

st.markdown("---")
st.caption("Data: xaus.com + Yahoo Finance • News: Kitco RSS • Alpaca Paper • Educational use only • Not financial advice")
