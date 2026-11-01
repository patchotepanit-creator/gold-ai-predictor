import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Gold AI Trader", page_icon=":bar_chart:", layout="wide")
st.title("AI Gold Day-to-Day Trader")
st.markdown("**Educational tool only — not financial advice.**")

mode = st.sidebar.radio("Mode", ["Daily Forecast", "Day Trading"])

if mode == "Daily Forecast":
    try:
        r = requests.get("https://xaus.com/api/v1/spot", timeout=10)
        price = r.json().get("spot_usd_oz") or r.json().get("xau", {}).get("price")
        st.metric("Current Gold Price", f"${float(price):,.2f}" if price else "N/A")
    except Exception as e:
        st.error(f"Price error: {e}")
else:
    try:
        data = yf.download("GC=F", period="5d", interval="15m", progress=False, auto_adjust=True)
        if not data.empty:
            last = float(data["Close"].iloc[-1])
            st.metric("Gold Futures (last 15m)", f"${last:,.2f}")
            st.line_chart(data["Close"].tail(50))
        else:
            st.warning("No intraday data")
    except Exception as e:
        st.error(f"Intraday error: {e}")

st.caption("Basic version — more features will be restored once this loads cleanly.")
