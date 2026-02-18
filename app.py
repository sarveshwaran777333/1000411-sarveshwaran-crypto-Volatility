import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os
from datetime import datetime, timedelta

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Crypto Volatility Visualizer", 
    layout="wide",
    page_icon="💰",
    initial_sidebar_state="expanded"
)

# 2. SESSION STATE & THEME LOGIC
if 'bg_color' not in st.session_state:
    st.session_state.bg_color = '#FFFFFF'
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_theme_colors(hex_color):
    """Calculates text color AND sidebar color based on main background"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    
    # 1. Determine Text Color
    if luminance > 160:
        text_color = '#000000'
        expander_bg = 'rgba(255, 255, 255, 0.4)'
    else:
        text_color = '#FFFFFF'
        expander_bg = 'rgba(0, 0, 0, 0.4)'
    
    # 2. Determine Sidebar Color
    if luminance > 160:
        sb_r, sb_g, sb_b = max(0, r-15), max(0, g-15), max(0, b-15)
    else:
        sb_r, sb_g, sb_b = min(255, r+20), min(255, g+20), min(255, b+20)
    
    sidebar_color = f"#{sb_r:02x}{sb_g:02x}{sb_b:02x}"
    
    return text_color, sidebar_color, expander_bg

text_color, sidebar_bg, expander_header_bg = get_theme_colors(st.session_state.bg_color)
grid_color = text_color 

# 3. ROBUST CSS INJECTION
st.markdown(
    f"""
    <style>
    /* Main App Background */
    .stApp {{
        background-color: {st.session_state.bg_color};
    }}
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
    }}
    
    /* GLOBAL TEXT COLOR FORCE */
    h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown, .stMetricValue, .stMetricLabel {{
        color: {text_color} !important;
    }}
    
    /* INPUTS: Force text color inside inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] span {{
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
    }}
    
    /* --- EXPANDER FIX (FROSTED GLASS LOOK) --- */
    .streamlit-expanderHeader {{
        background-color: {expander_header_bg} !important;
        border-radius: 5px;
        color: {text_color} !important;
    }}
    .streamlit-expanderHeader p, .streamlit-expanderHeader span {{
        color: {text_color} !important;
    }}
    .streamlit-expanderHeader svg {{
        fill: {text_color} !important;
    }}
    .streamlit-expanderHeader:hover {{
        filter: brightness(1.1);
        color: {text_color} !important;
    }}
    
    /* Dropdown Menu Items (Always black for readability) */
    ul[data-testid="stSelectboxVirtualDropdown"] li {{
        color: black !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# 4. SIDEBAR CONTROLS
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # --- GEMINI API KEY ---
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        api_key_configured = True
        st.success("AI Active & Ready! 🤖")
    else:
        api_key_configured = False
        st.error("⚠️ API Key missing in Secrets!")
    
    st.markdown("---")
    
    # NOTE: File Uploader Removed. App now auto-loads 'crypto_Currency_data.csv'
    
    with st.expander("🎨 Appearance", expanded=False):
        st.color_picker("Background Color", key="bg_color")
        
    with st.expander("🌊 Pattern Lab", expanded=False):
        pattern_type = st.selectbox("Wave Type", ("Sine Wave", "Cosine Wave", "Random Noise"))
        amplitude = st.slider("Amplitude", 10, 200, 50)
        frequency = st.slider("Frequency", 1, 100, 10)
        drift = st.slider("Trend Drift", -5, 5, 0)

    with st.expander("🧠 Strategy Params", expanded=True):
        fast_window = st.slider("Fast MA", 5, 50, 10)
        slow_window = st.slider("Slow MA", 20, 200, 50)

    with st.expander("🔮 Forecast Params", expanded=False):
        volatility = st.slider("Exp. Volatility (σ)", 0.01, 0.50, 0.20)
        expected_return = st.slider("Exp. Return (μ)", -0.50, 0.50, 0.10)
        time_steps = st.number_input("Days to Forecast", value=365)

# 5. APP LOGIC
st.title("💰 Crypto Volatility Visualizer")
st.markdown("Merged Framework: **Market Simulator** | **Live Strategy Backtester** | **Stochastic Forecasting** | **AI Chat**")

def simulate_gbm(mu, sigma, days, start_price=1000):
    dt = 1/365
    returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, days))
    return start_price * np.cumprod(returns)

def generate_signals(df, fast, slow):
    # 1. Moving Averages
    df['Fast_MA'] = df['Price'].rolling(window=fast).mean()
    df['Slow_MA'] = df['Price'].rolling(window=slow).mean()
    df['Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, -1)
    df['Entry_Exit'] = df['Signal'].diff()
    
    # 2. Performance Calculation (Cumulative Returns)
    df['Market_Ret'] = df['Price'].pct_change()
    df['Strat_Ret'] = df['Market_Ret'] * df['Signal'].shift(1)
    
    df['Cum_Market'] = (1 + df['Market_Ret']).cumprod()
    df['Cum_Strat'] = (1 + df['Strat_Ret']).cumprod()
    df.fillna(1, inplace=True)
    
    return df

# Helper to get AI Response
def ask_gemini_chat(chat_history, context=""):
    if not api_key_configured:
        return "⚠️ API Key missing. Please check your secrets."
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        system_prompt = f"""
        You are "Nexus", a friendly and enthusiastic Crypto AI Assistant.
        You are chatting with a user who is looking at a financial dashboard.
        
        CURRENT DASHBOARD CONTEXT:
        {context}
        
        GUIDELINES:
        - Be conversational, encouraging, and clear.
        - Use emojis 🚀📈 to make it fun.
        - Keep answers concise but helpful.
        """
        
        full_prompt = system_prompt + "\n\n"
        for msg in chat_history[-5:]:
             full_prompt += f"{msg['role'].upper()}: {msg['content']}\n"
             
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["🌊 Pattern Lab", "🧠 Real Data & Analysis", "🔮 Monte Carlo Forecast", "💬 AI Chat"])

# --- TAB 1: PATTERN LAB ---
with tab1:
    t = np.linspace(0, 100, 500)
    if "Sine" in pattern_type:
        base = amplitude * np.sin(frequency * 0.1 * t)
    elif "Cosine" in pattern_type:
        base = amplitude * np.cos(frequency * 0.1 * t)
    else:
        base = np.random.normal(0, amplitude, len(t))
    
    final_price = 1000 + base + (drift * t) + np.random.normal(0, amplitude * 0.2, len(t))
    
    fig = px.line(x=t, y=final_price, title=f"Generated {pattern_type}")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=text_color),
                      xaxis=dict(showgrid=False, title_font=dict(color=text_color), tickfont=dict(color=text_color)), 
                      yaxis=dict(showgrid=True, gridcolor=grid_color, gridwidth=0.1, title_font=dict(color=text_color), tickfont=dict(color=text_color)))
    fig.update_traces(line_color='#0068C9')
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: REAL DATA ---
with tab2:
    @st.cache_data
    def load_data():
        df = None
        
        # 1. Try Loading Local File
        if os.path.exists("crypto_Currency_data.csv"):
            try:
                df = pd.read_csv("crypto_Currency_data.csv")
                # st.toast("Dataset Loaded Successfully! 📂") # Optional user feedback
            except Exception as e:
                st.error(f"Error reading local file: {e}")
        
        # 2. If loaded, Process it
        if df is not None:
            try:
                # Handle Timestamp
                if 'Timestamp' in df.columns:
                    if df['Timestamp'].dtype == 'float64' or df['Timestamp'].dtype == 'int64':
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
                    else:
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                else:
                    date_col = [c for c in df.columns if 'date' in c.lower()]
                    if date_col:
                        df['Timestamp'] = pd.to_datetime(df[date_col[0]])
                    else:
                        df['Timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq='D')

                # Rename Close to Price
                if 'Close' in df.columns:
                    df.rename(columns={'Close': 'Price'}, inplace=True)
                
                # Check for Price column
                if 'Price' not in df.columns:
                    st.error("Dataset missing 'Close' or 'Price' column.")
                    return None
                
                # Ensure High/Low/Volume
                if 'High' not in df.columns: df['High'] = df['Price']
                if 'Low' not in df.columns: df['Low'] = df['Price']
                if 'Volume' not in df.columns: df['Volume'] = 0
                
                # Fill Missing
                df.fillna(method='ffill', inplace=True)
                df.dropna(inplace=True)
                
                return df
            except Exception as e:
                st.error(f"Data processing error: {e}")
                return None
        
        # 3. Fallback: Simulation (If no file exists)
        st.warning("⚠️ 'crypto_Currency_data.csv' not found. Using simulation data.")
        dates = pd.date_range(end=datetime.now(), periods=1000, freq="h")
        base_price = 40000 + np.cumsum(np.random.randn(1000)) * 100
        
        return pd.DataFrame({
            'Timestamp': dates,
            'Price': base_price,
            'High': base_price + 50,
            'Low': base_price - 50,
            'Volume': np.random.randint(100, 1000, 1000)
        })

    real_df = load_data()
    
    if real_df is not None:
        strategy_df = generate_signals(real_df.copy(), fast_window, slow_window)
        
        # METRICS
        daily_volatility = strategy_df['Price'].pct_change().std() * 100
        avg_drift = strategy_df['Price'].pct_change().mean() * 100
        
        c1, c2, c3, c4 = st.columns(4)
        curr_price = strategy_df['Price'].iloc[-1]
        signal_status = "BULLISH 🟢" if strategy_df['Signal'].iloc[-1] == 1 else "BEARISH 🔴"
        
        c1.metric("Current Price", f"${curr_price:,.2f}")
        c2.metric("Signal Status", signal_status)
        c3.metric("Volatility Index", f"{daily_volatility:.2f}%")
        c4.metric("Avg Drift", f"{avg_drift:.4f}%")

        # --- GRAPH 1: PRICE & SIGNALS ---
        st.subheader("1. Price Overview & Signals")
        fig_strat = go.Figure()
        fig_strat.add_trace(go.Scatter(x=strategy_df['Timestamp'], y=strategy_df['Price'], name="Price", line=dict(color='gray', width=1, dash='dot')))
        fig_strat.add_trace(go.Scatter(x=strategy_df['Timestamp'], y=strategy_df['Fast_MA'], name="Fast MA", line=dict(color='#00CFBE', width=2)))
        fig_strat.add_trace(go.Scatter(x=strategy_df['Timestamp'], y=strategy_df['Slow_MA'], name="Slow MA", line=dict(color='#FF4B4B', width=2)))
        
        buys = strategy_df[strategy_df['Entry_Exit'] == 2]
        sells = strategy_df[strategy_df['Entry_Exit'] == -2]
        fig_strat.add_trace(go.Scatter(x=buys['Timestamp'], y=buys['Price'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=15, color='#00FF00')))
        fig_strat.add_trace(go.Scatter(x=sells['Timestamp'], y=sells['Price'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=15, color='#FF0000')))

        fig_strat.update_layout(hovermode="x unified", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color=text_color), xaxis=dict(showgrid=False, title_font=dict(color=text_color), tickfont=dict(color=text_color)), 
                                yaxis=dict(showgrid=True, gridcolor=grid_color, gridwidth=0.1, title_font=dict(color=text_color), tickfont=dict(color=text_color)))
        st.plotly_chart(fig_strat, use_container_width=True)

        # --- GRAPH 2: HIGH vs LOW ---
        st.subheader("2. High vs Low Analysis")
        fig_hl = go.Figure()
        fig_hl.add_trace(go.Scatter(x=strategy_df['Timestamp'], y=strategy_df['High'], name="High", line=dict(color='#00FF00', width=1)))
        fig_hl.add_trace(go.Scatter(x=strategy_df['Timestamp'], y=strategy_df['Low'], name="Low", line=dict(color='#FF0000', width=1)))
        fig_hl.add_trace(go.Scatter(x=strategy_df['Timestamp'], y=strategy_df['Price'], name="Close", line=dict(color='white', width=1, dash='dot')))
        
        fig_hl.update_layout(hovermode="x unified", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color=text_color), xaxis=dict(showgrid=False, title_font=dict(color=text_color), tickfont=dict(color=text_color)), 
                                yaxis=dict(showgrid=True, gridcolor=grid_color, gridwidth=0.1, title_font=dict(color=text_color), tickfont=dict(color=text_color)))
        st.plotly_chart(fig_hl, use_container_width=True)

        # --- GRAPH 3: VOLUME ANALYSIS ---
        st.subheader("3. Volume Analysis")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=strategy_df['Timestamp'], y=strategy_df['Volume'], name="Volume", marker_color='#AB63FA'))
        
        fig_vol.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color=text_color), xaxis=dict(showgrid=False, title_font=dict(color=text_color), tickfont=dict(color=text_color)), 
                                yaxis=dict(showgrid=True, gridcolor=grid_color, gridwidth=0.1, title_font=dict(color=text_color), tickfont=dict(color=text_color)))
        st.plotly_chart(fig_vol, use_container_width=True)

        # --- GRAPH 4: EQUITY CURVE ---
        st.subheader("4. Strategy vs. Buy & Hold")
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=strategy_df['Timestamp'], y=strategy_df['Cum_Strat'], name="Strategy Return", line=dict(color='#00FF00', width=2), fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'))
        fig_perf.add_trace(go.Scatter(x=strategy_df['Timestamp'], y=strategy_df['Cum_Market'], name="Buy & Hold", line=dict(color='gray', width=2, dash='dot')))
        
        fig_perf.update_layout(hovermode="x unified", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font=dict(color=text_color), xaxis=dict(showgrid=False, title_font=dict(color=text_color), tickfont=dict(color=text_color)), 
                               yaxis=dict(title="Growth Multiplier (1.0 = Breakeven)", showgrid=True, gridcolor=grid_color, gridwidth=0.1, title_font=dict(color=text_color), tickfont=dict(color=text_color)))
        st.plotly_chart(fig_perf, use_container_width=True)

# --- TAB 3: FORECAST ---
with tab3:
    st.subheader("Stochastic Price Projection (GBM)")
    sim_prices = simulate_gbm(expected_return, volatility, int(time_steps), start_price=real_df['Price'].iloc[-1])
    sim_dates = [datetime.now() + timedelta(days=i) for i in range(len(sim_prices))]
    
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(x=sim_dates, y=sim_prices, name="Projected Path", line=dict(color='#AB63FA')))
    
    fig_mc.update_layout(title=f"Projected {time_steps} Day Path", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font=dict(color=text_color), xaxis=dict(showgrid=False, title_font=dict(color=text_color), tickfont=dict(color=text_color)), 
                         yaxis=dict(showgrid=True, gridcolor=grid_color, gridwidth=0.1, title_font=dict(color=text_color), tickfont=dict(color=text_color)))
    st.plotly_chart(fig_mc, use_container_width=True)

# --- TAB 4: CHAT INTERFACE ---
with tab4:
    st.subheader("💬 Chat with Nexus AI")
    
    # 1. Prepare Live Context
    context_data = f"""
    Current Market Price: ${curr_price:.2f}
    Current Signal: {signal_status}
    Fast MA Setting: {fast_window}
    Slow MA Setting: {slow_window}
    Forecasted Volatility: {volatility}
    """

    # 2. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Chat Input
    if prompt := st.chat_input("Ask me about the market..."):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = ask_gemini_chat(st.session_state.messages, context=context_data)
                st.markdown(response_text)
                
        # Add assistant message to state
        st.session_state.messages.append({"role": "assistant", "content": response_text})
