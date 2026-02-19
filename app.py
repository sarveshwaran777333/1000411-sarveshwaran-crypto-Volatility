import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import os
from datetime import datetime, timedelta

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Crypto Volatility Hub",
    layout="wide",
    page_icon="💰"
)

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # NEW AI CONFIGURATION STYLE
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        api_key_active = True
        st.success("AI Neural Link: Online 🤖")
    else:
        api_key_active = False
        st.warning("AI Key Missing in Secrets")

    st.markdown("---")
    st.subheader("📈 Strategy Parameters")
    fast_window = st.slider("Fast Moving Average", 5, 30, 10)
    slow_window = st.slider("Slow Moving Average", 40, 150, 60)
    
    st.markdown("---")
    st.subheader("🌊 Math Wave Params")
    amp = st.slider("Amplitude (Swing)", 10, 500, 100)
    freq = st.slider("Frequency (Speed)", 1, 50, 10)
    drift_val = st.slider("Market Drift (Trend)", -5.0, 5.0, 0.5)

# -------------------- FUNCTIONS --------------------

@st.cache_data
def load_and_clean():
    file_path = "crypto_Currency_data.csv"
    if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
        df = pd.read_csv(file_path)
    else:
        dates = pd.date_range(end=datetime.now(), periods=2000, freq="h")
        prices = 45000 + np.cumsum(np.random.randn(2000) * 150)
        df = pd.DataFrame({"Timestamp": dates, "Price": prices, "High": prices+100, "Low": prices-100, "Volume": np.random.randint(500, 5000, 2000)})

    if "Close" in df.columns: df.rename(columns={"Close": "Price"}, inplace=True)
    if "Timestamp" in df.columns:
        if df["Timestamp"].dtype in ['float64', 'int64']:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')
        else:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    
    df.ffill(inplace=True)
    return df.tail(1500).copy()

# NEW CHAT FUNCTION WITH YOUR RULES
def ask_ai(history):
    if not api_key_active: return "Please add GEMINI_API_KEY to secrets."
    
    # Using the model and prompt style you provided
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    SYSTEM_PROMPT = """
    You are Nexus, a crypto-only AI assistant.
    Rules:
    - Answer ONLY crypto, market, and dashboard questions.
    - Use simple English.
    - Maximum 5 lines.
    - If question is not about crypto/markets, reply:
    "I can help only with crypto and market analysis questions."
    """
    
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser History:\n"
    for m in history[-5:]:
        full_prompt += f"{m['role']}: {m['content']}\n"
    
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------- INTERFACE --------------------

st.title("💰 Crypto Volatility Visualizer")
tabs = st.tabs(["🌊 Pattern Lab", "🧠 Real Analysis", "🔮 Forecast", "💬 AI Assistant"])

# TAB 1: MATH
with tabs[0]:
    t = np.linspace(0, 20, 1000)
    wave_price = 1000 + amp * np.sin(freq * 0.1 * t) + (drift_val * 50 * t)
    fig_math = go.Figure(go.Scattergl(y=wave_price, name="Math Wave", line=dict(color='#00CFBE')))
    st.plotly_chart(fig_math, use_container_width=True)

# TAB 2: DATA
with tabs[1]:
    df = load_and_clean()
    df["Fast_MA"] = df["Price"].rolling(fast_window).mean()
    df["Slow_MA"] = df["Price"].rolling(slow_window).mean()
    
    f1 = go.Figure()
    f1.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Price"], name="Price"))
    f1.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Fast_MA"], name="Fast MA"))
    f1.add_trace(go.Scattergl(x=df["Timestamp"], y=df["Slow_MA"], name="Slow MA"))
    st.plotly_chart(f1, use_container_width=True)

# TAB 4: CHAT (Applying your new logic)
with tabs[3]:
    st.subheader("💬 Nexus Analyst AI")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
        
    if p := st.chat_input("Ask me about the market..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                resp = ask_ai(st.session_state.messages)
                st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
