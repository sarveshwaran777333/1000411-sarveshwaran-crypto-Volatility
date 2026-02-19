import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime

# -------------------- CONFIG & API SETUP --------------------
st.set_page_config(page_title="Crypto Analyst Hub", layout="wide", page_icon="💰")

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🎛️ Dashboard Controls")
    st.subheader("📈 Moving Averages")
    fast_window = st.slider("Fast MA Period", 5, 50, 10)
    slow_window = st.slider("Slow MA Period", 20, 200, 50)
    st.markdown("---")

# -------------------- DATA PROCESSING --------------------
@st.cache_data
def load_data():
    file_path = "crypto_Currency_data.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if "Close" in df.columns: df.rename(columns={"Close": "Price"}, inplace=True)
            if "Timestamp" in df.columns:
                if df["Timestamp"].dtype != 'O':
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')
                else:
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df.ffill(inplace=True)
            return df.tail(1500).copy()
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None
    return None

df = load_data()

# -------------------- ANALYTICS LOGIC --------------------
if df is not None:
    df['Fast_MA'] = df['Price'].rolling(fast_window).mean()
    df['Slow_MA'] = df['Price'].rolling(slow_window).mean()
    rets = df['Price'].pct_change()
    vol = rets.rolling(20).std()
    df['Regime'] = np.where(vol > vol.median(), 'Volatile', 'Stable')

# -------------------- DASHBOARD UI --------------------
st.title("💰 Crypto Volatility")

if df is None:
    st.error("❌ Data file `crypto_Currency_data.csv` not found.")
    st.info("Please upload the CSV to your GitHub folder.")
else:
    tab1 = st.tabs(["📈 Market Analytics"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Price Trend")
            fig1 = go.Figure()
            fig1.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Price"], name="Price", line=dict(color="#00CFBE")))
            fig1.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Fast_MA"], name="Fast MA", line=dict(color="orange")))
            fig1.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Slow_MA"], name="Slow MA", line=dict(color="red")))
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("2. Trading Volume")
            fig2 = go.Figure(go.Bar(x=df["Timestamp"], y=df["Volume"], marker_color="#AB63FA"))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.subheader("3. Day High/Low")
            fig3 = go.Figure()
            fig3.add_trace(go.Scattergl(x=df["Timestamp"], y=df["High"], name="High", line=dict(color="green")))
            fig3.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Low"], name="Low", line=dict(color="red")))
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("4. Volatility Regimes")
            fig4 = go.Figure()
            stable = df[df['Regime'] == 'Stable']
            volat = df[df['Regime'] == 'Volatile']
            fig4.add_trace(go.Scattergl(x=stable["Timestamp"], y=stable["Price"], mode='markers', name="Stable", marker=dict(color="blue", size=4)))
            fig4.add_trace(go.Scattergl(x=volat["Timestamp"], y=volat["Price"], mode='markers', name="Volatile", marker=dict(color="orange", size=4)))
            st.plotly_chart(fig4, use_container_width=True)

    
