# 🥇 AI Gold Trading Price Predictor

A simple Streamlit web app that uses machine learning (Random Forest) to predict short-term gold prices based on historical technical features.

## Features
- Live / latest gold spot price
- Historical price chart
- AI-powered next-day (or few days) price prediction
- Bullish / Bearish signal
- Model performance metrics (MAE, R²)
- Educational disclaimer

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Open the URL shown in the terminal (usually http://localhost:8501)

## Data Source
- Free public API: [xaus.com](https://xaus.com/api/) (no API key required)
- Endpoints used: `/api/v1/history` and `/api/v1/spot`

## Disclaimer
This is an **educational prototype only**. It is **not financial advice**. 
Gold price prediction is extremely difficult. Do not trade based solely on this model. 
Always perform your own research and consider consulting a financial advisor.

## Model
- Features: moving averages (5/10/20), RSI, volatility, lags, high-low spread, returns
- Algorithm: Random Forest Regressor
- Target: future close price (configurable horizon)
- Simple 80/20 train/test split for evaluation

You can improve it by adding more features (macro data, DXY, silver, sentiment), using walk-forward validation, LSTM/Transformers, or better risk management.
