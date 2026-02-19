import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import os
from datetime import datetime, timedelta

# -------------------- 1. CONFIG & API SETUP --------------------
st.set_page_config(page_title="Crypto Analyst Hub", layout="wide", page_icon="💰")

# Check for the API Key (using the name from your farming app)
api_key = st.secrets.get("GENAI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    # Using 1.5-flash as it is highly stable for general API keys
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    api_ready = True
else:
    api_ready = False

# -------------------- 2. SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- 3. SIDEBAR CONTROLS --------------------
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    st.subheader("📈 Strategy Parameters")
    fast_window = st.slider("Fast Moving Average", 5, 50, 10)
    slow_window = st.slider("Slow Moving Average", 20, 200, 50)
    
    st.markdown("---")
    st.subheader("🌊 Math Simulation Params")
    amp = st.slider("Amplitude (Swing)", 10, 500, 100)
    freq = st.slider("Frequency (Speed)", 1, 50, 10)
    drift_val = st.slider("Market Drift", -5.0, 5.0, 0.5)

# -------------------- 4. DATA PROCESSING --------------------
@st.cache_data
def load_data():
    file_path = "crypto_Currency_data.csv"
    
    # Try to load the real file first (Checks if it's not a tiny Git LFS pointer)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 2000:
        try:
            df = pd.read_csv(file_path)
            # Standardize
            if "Close" in df.columns: df.rename(columns={"Close": "Price"}, inplace=True)
            if "Timestamp" in df.columns:
                # Handle Unix timestamps 
                if df["Timestamp"].dtype != 'O':
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')
                else:
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            return df.tail(1000).copy()
        except Exception as e:
            st.warning(f"Failed to read CSV, using simulation. Error: {e}")
            pass 

    # FALLBACK: Mathematical Simulation (Satisfies Assignment Requirements)
    st.toast("Using Math-Simulated Data", icon="🔢")
    t = np.linspace(0, 20, 1000)
    # Math formula: Base + Sine Wave + Trend Drift + Random Noise
    base_price = 40000 + amp * np.sin(freq * 0.1 * t) + (drift_val * 50 * t)
    noise = np.random.normal(0, amp * 0.2, 1000)
    sim_price = base_price + noise
    
    dates = pd.date_range(end=datetime.now(), periods=1000, freq="h")
    df_sim = pd.DataFrame({
        "Timestamp": dates,
        "Price": sim_price,
        "High": sim_price + np.random.uniform(50, 150, 1000),
        "Low": sim_price - np.random.uniform(50, 150, 1000),
        "Volume": np.random.randint(1000, 10000, 1000)
    })
    return df_sim

df = load_data()

# Calculate Moving Averages & Regimes
df['Fast_MA'] = df['Price'].rolling(fast_window).mean()
df['Slow_MA'] = df['Price'].rolling(slow_window).mean()

rets = df['Price'].pct_change()
vol = rets.rolling(20).std()
df['Regime'] = np.where(vol > vol.median(), 'Volatile', 'Stable')

def get_nexus_response(user_input):
    if not api_ready:
        return "❌ API key not configured properly in Streamlit Secrets."

    SYSTEM_PROMPT = """
    You are Nexus, a crypto-only AI assistant.
    Rules:
    - Answer ONLY crypto, market, and dashboard questions.
    - Use simple English.
    - Maximum 5 lines.
    - If question is not about crypto/markets, reply:
    "I can help only with crypto and market analysis questions."
    """

    try:
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser Question: {user_input}"
        
        # ✅ Use 'model.generate_content' with config as a simple dict
        response = model.generate_content(
            full_prompt,
            config={
                "temperature": 0.7,
                "max_output_tokens": 500
            }
        )

        # SAFE extraction for 2.5 Flash
        if response:
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content.parts:
                    return candidate.content.parts[0].text.strip()

        return "⚠️ AI returned empty content. Try again."

    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"


# -------------------- 6. DASHBOARD UI --------------------
st.title("💰 Crypto Volatility & AI Analyst")

tab1, tab2 = st.tabs(["📈 Market Analytics", "💬 AI Assistant"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart 1: Price Trend & Moving Averages
        st.subheader("1. Price Trend & Moving Averages")
        fig1 = go.Figure()
        fig1.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Price"], name="Price", line=dict(color="#00CFBE")))
        fig1.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Fast_MA"], name=f"{fast_window} MA", line=dict(color="orange")))
        fig1.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Slow_MA"], name=f"{slow_window} MA", line=dict(color="red")))
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Volume Bar Chart
        st.subheader("2. Trading Volume")
        fig2 = go.Figure(go.Bar(x=df["Timestamp"], y=df["Volume"], marker_color="#AB63FA"))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Chart 3: High vs Low Spread
        st.subheader("3. Intra-day Volatility (High/Low)")
        fig3 = go.Figure()
        fig3.add_trace(go.Scattergl(x=df["Timestamp"], y=df["High"], name="High", line=dict(color="green")))
        fig3.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Low"], name="Low", line=dict(color="red")))
        st.plotly_chart(fig3, use_container_width=True)

        # Chart 4: Stable vs Volatile Regimes
        st.subheader("4. Stability vs Volatility Regimes")
        fig4 = go.Figure()
        stable = df[df['Regime'] == 'Stable']
        volat = df[df['Regime'] == 'Volatile']
        fig4.add_trace(go.Scattergl(x=stable["Timestamp"], y=stable["Price"], mode='markers', name="Stable", marker=dict(color="blue", size=4)))
        fig4.add_trace(go.Scattergl(x=volat["Timestamp"], y=volat["Price"], mode='markers', name="Volatile", marker=dict(color="orange", size=4)))
        st.plotly_chart(fig4, use_container_width=True)

# -------------------- AI Chat in tab2 --------------------
with tab2:
    st.subheader("Chat with Nexus 🤖")

    if not api_ready:
        st.error("❌ API Key not found! Please add 'GENAI_API_KEY' or 'GEMINI_API_KEY' to your Streamlit Secrets.")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about crypto volatility..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_nexus_response(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

