import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Volatility Visualizer", layout="wide")

st.title("💰Crypto Volatility Visualizer")
st.markdown("Compare **Real Market Data Visualization** with **Mathematical Simulations**")

st.sidebar.header("User Controls")

pattern_type = st.sidebar.selectbox(
    "Select Simulation Pattern",
    ("Sine Wave (Cycles)", "Cosine Wave (Cycles)", "Random Noise (Shocks)")
)

amplitude = st.sidebar.slider("Amplitude (Risk/Swing Size)", 10, 200, 50)
frequency = st.sidebar.slider("Frequency (Speed of Swings)", 1, 100, 10)
drift = st.sidebar.slider("Drift (Long-term Trend)", -5, 5, 0)

show_real_data = st.sidebar.checkbox("Show Real Bitcoin Data", value=True)

st.header("Market Simulator")
st.info(f"Visualizing: **{pattern_type}** with Amplitude={amplitude}, Freq={frequency}, Drift={drift}")

t = np.linspace(0, 100, 500)

if "Sine" in pattern_type:
    base_price = amplitude * np.sin(frequency * 0.1 * t)
elif "Cosine" in pattern_type:
    base_price = amplitude * np.cos(frequency * 0.1 * t)
else:
    base_price = np.random.normal(0, amplitude, size=len(t))

trend = drift * t
noise = np.random.normal(0, amplitude * 0.2, size=len(t))
final_price = 1000 + base_price + trend + noise

sim_df = pd.DataFrame({"Date": t, "Price": final_price})

fig_sim = px.line(sim_df, x="Date", y="Price", title=f"Simulated {pattern_type} Price Action")
st.plotly_chart(fig_sim, use_container_width=True)

if show_real_data:
    st.header("Real Bitcoin Data Analysis")
   
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv("crypto_Currency_data.csv")
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
            df = df.dropna()
            df = df.rename(columns={'Close': 'Price'})
            return df.tail(500)
        except Exception as e:
            return None
       
    real_df = load_data()
   
    if real_df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${real_df['Price'].iloc[-1]:.2f}")
        col2.metric("Highest Price", f"${real_df['High'].max():.2f}")
        col3.metric("Lowest Price", f"${real_df['Low'].min():.2f}")
       
        st.subheader("Price Over Time")
        fig_real = px.line(real_df, x='Timestamp', y='Price', title="Bitcoin Closing Price")
        st.plotly_chart(fig_real, use_container_width=True)
       
        st.subheader("High vs Low Comparison")
        st.caption("This graph plots both High (Green) and Low (Red) prices on the same chart to show the volatility range.")
       
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=real_df['Timestamp'], y=real_df['High'],
            mode='lines', line_color='green', name='High Price'
        ))
        fig_vol.add_trace(go.Scatter(
            x=real_df['Timestamp'], y=real_df['Low'],
            mode='lines', line_color='red', name='Low Price',
            fill='tonexty'
        ))
        fig_vol.update_layout(title="High vs Low Prices (Volatility Channel)")
        st.plotly_chart(fig_vol, use_container_width=True)
           
        st.subheader("Trading Volume")
        fig_vol_bar = px.bar(real_df, x='Timestamp', y='Volume', title="Trading Volume")
        st.plotly_chart(fig_vol_bar, use_container_width=True)
       
    else:
        st.error("Error loading data. Please ensure 'btcusd_1-min_data.csv.crdownload' is in the same folder.")

st.markdown("---")
