import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# -------------------- 1. PAGE CONFIG --------------------
st.set_page_config(page_title="Crypto Analytics Dashboard", layout="wide", page_icon="💰")

# -------------------- 2. SIDEBAR CONTROLS --------------------
with st.sidebar:
    st.title("🎛️ Settings")
    st.subheader("📈 Moving Averages")
    fast_window = st.slider("Fast Period (Short-term)", 5, 50, 10)
    slow_window = st.slider("Slow Period (Long-term)", 20, 200, 50)
    st.markdown("---")
    st.caption("Data source: crypto_Currency_data.csv")

# -------------------- 3. DATA LOADING --------------------
@st.cache_data
def load_data():
    file_path = "crypto_Currency_data.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            # Standardize Column Names
            if "Close" in df.columns: df.rename(columns={"Close": "Price"}, inplace=True)
            
            # Convert Timestamps
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

# -------------------- 4. MAIN INTERFACE --------------------
st.title("💰 Crypto Volatility Dashboard")

if df is None:
    st.error("❌ Data file `crypto_Currency_data.csv` not found.")
    st.info("Ensure the CSV file is uploaded to the same folder as this script.")
else:
    # --- ANALYTICS CALCULATIONS ---
    df['Fast_MA'] = df['Price'].rolling(fast_window).mean()
    df['Slow_MA'] = df['Price'].rolling(slow_window).mean()
    
    # Calculate Volatility Regimes (Rubric Requirement)
    rets = df['Price'].pct_change()
    vol = rets.rolling(20).std()
    df['Regime'] = np.where(vol > vol.median(), 'Volatile', 'Stable')

    # --- 2x2 CHART GRID ---
    col1, col2 = st.columns(2)

    with col1:
        # Chart 1: Price Trend
        st.subheader("1. Price Trend & MAs")
        fig1 = go.Figure()
        fig1.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Price"], name="Price", line=dict(color="#00CFBE")))
        fig1.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Fast_MA"], name=f"{fast_window} MA", line=dict(color="orange")))
        fig1.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Slow_MA"], name=f"{slow_window} MA", line=dict(color="red")))
        fig1.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Volume
        st.subheader("2. Trading Volume")
        fig2 = go.Figure(go.Bar(x=df["Timestamp"], y=df["Volume"], marker_color="#AB63FA"))
        fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Chart 3: High/Low Spread
        st.subheader("3. Day High/Low Range")
        fig3 = go.Figure()
        fig3.add_trace(go.Scattergl(x=df["Timestamp"], y=df["High"], name="High", line=dict(color="green")))
        fig3.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Low"], name="Low", line=dict(color="red")))
        fig3.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig3, use_container_width=True)

        # Chart 4: Volatility Regimes
        st.subheader("4. Stability vs Volatility")
        fig4 = go.Figure()
        stable = df[df['Regime'] == 'Stable']
        volat = df[df['Regime'] == 'Volatile']
        fig4.add_trace(go.Scattergl(x=stable["Timestamp"], y=stable["Price"], mode='markers', name="Stable", marker=dict(color="blue", size=4)))
        fig4.add_trace(go.Scattergl(x=volat["Timestamp"], y=volat["Price"], mode='markers', name="Volatile", marker=dict(color="orange", size=4)))
        fig4.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig4, use_container_width=True)
