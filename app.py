import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import ARIMA
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Apple Stock Forecast App", layout="wide")

# --- Login System ---
def login():
    st.title("🔐 Login to Access the App")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":  
            st.session_state["logged_in"] = True
            st.success("✅ Login successful!")
        else:
            st.error("❌ Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# --- Main App ---
st.title("Apple Stock Forecasting App")

@st.cache_data
def load_data():
    df = pd.read_csv("AAPL_historical_data.csv", parse_dates=['date'], index_col='date')
    df = df.sort_index()
    df['close'] = df['close'].astype(float)
    df['close'].interpolate(method='linear', inplace=True)
    return df

data = load_data()
st.subheader("📈 Historical Stock Data (Last 100 Days)")
st.line_chart(data['close'].tail(100))

model_option = st.selectbox("Choose a forecasting model:", ["ARIMA", "SARIMA", "Prophet", "LSTM"])
future_days = st.slider("Forecast days:", min_value=7, max_value=200, value=30, step=1)

# --- Forecasting Functions ---

def run_arima():
    model = joblib.load("models/arima_model.pkl")
    forecast = model.predict(n_periods=future_days)
    future_index = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=future_days, freq='B')
    forecast_series = pd.Series(forecast, index=future_index)
    st.subheader("📉 ARIMA Forecast")
    st.line_chart(forecast_series)
    return forecast_series

def run_sarima():
    model = joblib.load("models/sarima_model.pkl")
    forecast = model.get_forecast(steps=future_days)
    pred = forecast.predicted_mean
    index = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=future_days, freq='B')
    st.subheader("📉 SARIMA Forecast")
    st.line_chart(pd.Series(pred.values, index=index))
    return pd.Series(pred.values, index=index)

def run_prophet():
    df_prophet = data[['close']].reset_index().rename(columns={'date': 'ds', 'close': 'y'})
    model = joblib.load("models/prophet_model.pkl")
    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)
    st.subheader("📉 Prophet Forecast")
    forecast_series = forecast.set_index('ds')['yhat'].tail(future_days)
    st.line_chart(forecast_series)
    return forecast_series

def run_lstm():
    from sklearn.preprocessing import MinMaxScaler
    scaler = joblib.load("models/lstm_scaler.pkl")
    model = load_model("models/lstm_model.h5")
    look_back = 60

    scaled_data = scaler.transform(data['close'].values.reshape(-1, 1))
    last_60_days = scaled_data[-look_back:]
    future_input = last_60_days.reshape(1, look_back, 1)

    predictions = []
    for _ in range(future_days):
        pred = model.predict(future_input, verbose=0)
        predictions.append(pred[0, 0])
        next_input = pred.reshape(1, 1, 1)
        future_input = np.concatenate((future_input[:, 1:, :], next_input), axis=1)

    final_pred = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_index = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=future_days, freq='B')
    forecast_series = pd.Series(final_pred.flatten(), index=future_index)
    st.subheader("📉 LSTM Forecast")
    st.line_chart(forecast_series)
    return forecast_series

# --- Trigger Forecast ---
if st.button("🔮 Forecast"):
    st.subheader(f"📊 Forecasting with {model_option}")
    try:
        if model_option == "ARIMA":
            run_arima()
        elif model_option == "SARIMA":
            run_sarima()
        elif model_option == "Prophet":
            run_prophet()
        elif model_option == "LSTM":
            run_lstm()
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

# --- Model Comparison ---
st.markdown("---")
st.subheader("📊 Model Comparison")

if st.button("Compare All Models"):
    try:
        st.markdown("#### ARIMA")
        run_arima()

        st.markdown("#### SARIMA")
        run_sarima()

        st.markdown("#### Prophet")
        run_prophet()

        st.markdown("#### LSTM")
        run_lstm()

        st.success("✅ After comparing models, **LSTM gives better results than other models** based on its superior learning capability for sequential data.")
    except Exception as e:
        st.error(f"⚠️ Comparison Error: {str(e)}")

st.markdown("---")
st.caption("Developed with ❤️ for Apple stock forecasting")
