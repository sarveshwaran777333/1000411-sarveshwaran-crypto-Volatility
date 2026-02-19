import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import os
from datetime import datetime, timedelta

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Crypto Volatility Visualizer",
    layout="wide",
    page_icon="💰"
)

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False

curr_price = 0
signal_status = "N/A"
real_df = None

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🎛️ Control Panel")

    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        api_key_configured = True
        st.success("AI Active 🤖")
    else:
        api_key_configured = False
        st.warning("No Gemini API Key")

    st.markdown("---")

    fast_window = st.slider("Fast MA", 5, 50, 10)
    slow_window = st.slider("Slow MA", 20, 200, 50)

    volatility = st.slider("Exp. Volatility (σ)", 0.01, 0.50, 0.20)
    expected_return = st.slider("Exp. Return (μ)", -0.50, 0.50, 0.10)
    time_steps = st.number_input("Days to Forecast", min_value=1, value=365)

# -------------------- FUNCTIONS --------------------

@st.cache_data
def load_data():
    file_path = "crypto_Currency_data.csv"
    
    if os.path.exists(file_path):
        try:
            # DEBUG: See what the file looks like on the server
            file_size = os.path.getsize(file_path)
            
            # If the file is very small (like < 500 bytes), it's likely an LFS pointer
            if file_size < 500:
                st.error(f"⚠️ Detected a Git LFS pointer (Size: {file_size} bytes) instead of the real data. Please re-upload the CSV normally.")
                return None
                
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            st.error("❌ The CSV file is empty or formatted incorrectly.")
            return None
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return None
    else:
        # If file is totally missing, use simulation
        st.warning("📂 'crypto_Currency_data.csv' not found. Using simulation data.")
        dates = pd.date_range(end=datetime.now(), periods=1500, freq="h")
        base_price = 40000 + np.cumsum(np.random.randn(1500)) * 100
        df = pd.DataFrame({
            "Timestamp": dates,
            "Price": base_price,
            "High": base_price + 50,
            "Low": base_price - 50,
            "Volume": np.random.randint(100, 1000, 1500)
        })

    # ---- STANDARDIZE ----
    if "Close" in df.columns:
        df.rename(columns={"Close": "Price"}, inplace=True)

    if "Timestamp" in df.columns:
        # Handle Unix Timestamps (the format in your specific file)
        if df["Timestamp"].dtype in ['float64', 'int64']:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')
        else:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    else:
        df["Timestamp"] = pd.date_range(end=datetime.now(), periods=len(df))

    df.ffill(inplace=True)
    df.dropna(subset=['Price'], inplace=True) # Ensure we have prices

    # Keep only the last 1500 rows for high-speed charts
    return df.tail(1500).copy()

@st.cache_data
def generate_signals(df, fast, slow):
    df["Fast_MA"] = df["Price"].rolling(fast).mean()
    df["Slow_MA"] = df["Price"].rolling(slow).mean()

    df["Signal"] = np.where(df["Fast_MA"] > df["Slow_MA"], 1, -1)
    df["Entry_Exit"] = df["Signal"].diff()

    df["Market_Ret"] = df["Price"].pct_change()
    df["Strat_Ret"] = df["Market_Ret"] * df["Signal"].shift(1)

    df["Cum_Market"] = (1 + df["Market_Ret"]).cumprod()
    df["Cum_Strat"] = (1 + df["Strat_Ret"]).cumprod()

    # Create Volatility Regime Flag (for the 4th required chart)
    df['Rolling_Std'] = df['Market_Ret'].rolling(window=20).std()
    median_vol = df['Rolling_Std'].median()
    df['Regime'] = np.where(df['Rolling_Std'] > median_vol, 'Volatile', 'Stable')

    df.fillna(1, inplace=True)

    return df


def simulate_gbm(mu, sigma, days, start_price):
    dt = 1 / 365
    returns = np.exp(
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * np.random.normal(0, 1, days)
    )
    return start_price * np.cumprod(returns)


def ask_gemini(chat_history):
    if not api_key_configured:
        return "API Key missing."

    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-5:]])
    response = model.generate_content(prompt)
    return response.text


# -------------------- MAIN --------------------

st.title("💰 Crypto Volatility Visualizer (Optimized)")
st.markdown("Visualizing market swings, moving averages, and volatility regimes.")

real_df = load_data()

if real_df is not None:
    strategy_df = generate_signals(real_df.copy(), fast_window, slow_window)

    curr_price = strategy_df["Price"].iloc[-1]
    signal_status = "BULLISH 🟢" if strategy_df["Signal"].iloc[-1] == 1 else "BEARISH 🔴"
    
    # Safely calculate volatility avoiding NaN
    clean_returns = strategy_df['Market_Ret'].replace(1, np.nan).dropna()
    current_volatility = clean_returns.std() * 100 if not clean_returns.empty else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"${curr_price:,.2f}")
    c2.metric("Signal", signal_status)
    c3.metric("Daily Volatility", f"{current_volatility:.2f}%")

    st.markdown("---")

    # -------------------- GRAPH 1: PRICE LINE CHART --------------------
    st.subheader("📈 1. Trend Analysis (Price & Moving Averages)")
    st.markdown("*This line chart identifies overall market trends using Fast and Slow Moving Averages to generate bullish/bearish signals.*")
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scattergl(x=strategy_df["Timestamp"], y=strategy_df["Price"], name="Price", line=dict(color='gray', dash='dot')))
    fig1.add_trace(go.Scattergl(x=strategy_df["Timestamp"], y=strategy_df["Fast_MA"], name="Fast MA", line=dict(color='#00CFBE')))
    fig1.add_trace(go.Scattergl(x=strategy_df["Timestamp"], y=strategy_df["Slow_MA"], name="Slow MA", line=dict(color='#FF4B4B')))
    fig1.update_layout(height=400, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig1, use_container_width=True)

    # -------------------- GRAPH 2: HIGH-LOW COMPARISON --------------------
    st.subheader("↕️ 2. Intra-period Volatility (High vs Low)")
    st.markdown("*The gap between the High and Low prices highlights the maximum intra-period price swing, indicating sudden market stress or euphoria.*")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scattergl(x=strategy_df["Timestamp"], y=strategy_df["High"], name="High", line=dict(color='#00FF00', width=1)))
    fig2.add_trace(go.Scattergl(x=strategy_df["Timestamp"], y=strategy_df["Low"], name="Low", fill='tonexty', fillcolor='rgba(0,255,0,0.1)', line=dict(color='#FF0000', width=1)))
    fig2.update_layout(height=400, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------- GRAPH 3: VOLUME CHART --------------------
    st.subheader("📊 3. Trading Volume")
    st.markdown("*High trading volume often confirms a trend or precedes a breakout. Low volume suggests market indecision.*")
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=strategy_df["Timestamp"], y=strategy_df["Volume"], name="Volume", marker_color='#AB63FA'))
    fig3.update_layout(height=300, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig3, use_container_width=True)

    # -------------------- GRAPH 4: STABLE VS VOLATILE PERIODS --------------------
    st.subheader("⚖️ 4. Market Regimes (Stable vs. Volatile)")
    st.markdown("*By calculating a rolling standard deviation of returns, we can classify time periods into 'Stable' (blue) and 'Volatile' (orange) regimes. Volatile periods carry higher risk.*")
    
    stable_df = strategy_df[strategy_df['Regime'] == 'Stable']
    volatile_df = strategy_df[strategy_df['Regime'] == 'Volatile']

    fig4 = go.Figure()
    fig4.add_trace(go.Scattergl(x=stable_df["Timestamp"], y=stable_df["Price"], mode='markers', name="Stable Phase", marker=dict(color='#3498db', size=5)))
    fig4.add_trace(go.Scattergl(x=volatile_df["Timestamp"], y=volatile_df["Price"], mode='markers', name="Volatile Phase", marker=dict(color='#e67e22', size=5)))
    fig4.update_layout(height=400, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # -------------------- MONTE CARLO (Button Controlled) --------------------
    st.subheader("🔮 Monte Carlo Forecast")

    if st.button("Run Simulation"):
        st.session_state.simulation_run = True

    if st.session_state.simulation_run:
        sim_prices = simulate_gbm(
            expected_return,
            volatility,
            int(time_steps),
            curr_price
        )

        sim_dates = [
            datetime.now() + timedelta(days=i)
            for i in range(len(sim_prices))
        ]

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scattergl(
            x=sim_dates,
            y=sim_prices,
            name="Projection"
        ))
        
        fig_mc.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_mc, use_container_width=True)

# -------------------- CHAT --------------------
st.markdown("---")
st.subheader("💬 AI Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the market..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_gemini(st.session_state.messages)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
