import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import os
from datetime import datetime

# -------------------- CONFIG & API --------------------
st.set_page_config(page_title="Crypto Analyst Hub", layout="wide")

# IMPORTANT: Ensure your Secret is named "GENAI_API_KEY" or "GEMINI_API_KEY"
# I will check for both to be safe.
api_key = st.secrets.get("GENAI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    # Using 1.5-flash as it is more compatible with standard API keys than 2.0-preview
    model = genai.GenerativeModel("gemini-1.5-flash")
    api_ready = True
else:
    api_ready = False

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_data():
    file_path = "crypto_Currency_data.csv"
    if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
        df = pd.read_csv(file_path)
        if "Close" in df.columns: df.rename(columns={"Close": "Price"}, inplace=True)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s' if df["Timestamp"].dtype != 'O' else None)
        return df.tail(1000)
    return None

df = load_data()

# -------------------- CHAT LOGIC --------------------
def get_ai_response(user_input):
    if not api_ready:
        return "❌ API key not configured properly."

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
        full_message = f"{SYSTEM_PROMPT}\n\nUser Question: {user_input}"

        response = model.generate_content(full_message)

        # 🔥 SAFETY CHECKS
        if not response:
            return "⚠️ Empty response from AI."

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        if hasattr(response, "candidates") and response.candidates:
            return response.candidates[0].content.parts[0].text.strip()

        return "⚠️ AI returned no usable content."

    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"


# -------------------- UI LAYOUT --------------------
st.title("💰 Crypto Volatility & AI Analyst")

tab1, tab2 = st.tabs(["📈 Market Dashboard", "💬 AI Assistant"])

with tab1:
    if df is not None:
        st.subheader("Recent Market Activity")
        fig = go.Figure(go.Scattergl(x=df["Timestamp"], y=df["Price"], name="Price", line=dict(color="#00CFBE")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please ensure crypto_Currency_data.csv is in your GitHub repo.")

with tab2:
    st.subheader("Chat with Nexus")
    
    if not api_ready:
        st.error("❌ API Key not found! Please add 'GENAI_API_KEY' to your Streamlit Secrets.")
    
    # Display chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about crypto volatility..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_ai_response(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
