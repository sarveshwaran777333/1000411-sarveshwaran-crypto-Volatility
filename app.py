import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

def get_theme_colors(hex_color):
    """Calculates text color AND sidebar color based on main background"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    
    # Determine Text Color (Black or White)
    text_color = '#000000' if luminance > 128 else '#FFFFFF'
    
    # Determine Sidebar Color (Adjust brightness slightly for contrast)
    if luminance > 128:
        sb_r, sb_g, sb_b = max(0, r-15), max(0, g-15), max(0, b-15)
    else:
        sb_r, sb_g, sb_b = min(255, r+20), min(255, g+20), min(255, b+20)
    
    sidebar_color = f"#{sb_r:02x}{sb_g:02x}{sb_b:02x}"
    
    return text_color, sidebar_color

# Calculate colors immediately
text_color, sidebar_bg = get_theme_colors(st.session_state.bg_color)
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
    
    /* Global Text Color */
    h1, h2, h3, h4, h5, h6, p, li, span, div, label {{
        color: {text_color} !important;
    }}
    
    /* Fix Input Widgets Text (Sliders, Inputs) */
    .stSelectbox, .stSlider, .stNumberInput, .stDateInput {{
        color: {text_color} !important;
    }}
    div[data-testid="stMarkdownContainer"] p {{
        color: {text_color} !important;
    }}
    
    /* Metric styling */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
        color: {text_color} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# 4. SIDEBAR CONTROLS
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    with st.expander("🎨 Appearance", expanded=True):
        st.color_picker("Background Color", key="bg_color")
        st.caption("Sidebar auto-adjusts to match.")
        
    st.markdown("---")

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
st.markdown("Merged Framework: **Market Simulator** | **Live Strategy Backtester** | **Stochastic Forecasting**")

def simulate_gbm(mu, sigma, days, start_price=1000):
    dt = 1/365
    returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, days))
    return start_price * np.cumprod(returns)

def generate_signals(df, fast, slow):
    df['Fast_MA'] = df['Price'].rolling(window=fast).mean()
    df['Slow_MA'] = df['Price'].rolling(window=slow).mean()
    df['Signal'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, -1)
    df['Entry_Exit'] = df['Signal'].diff()
    return df

# TABS
tab1, tab2, tab3 = st.tabs(["🌊 Pattern Lab", "🧠 Strategy Backtester", "🔮 Monte Carlo Forecast"])

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
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color=text_color),
        xaxis=dict(showgrid=False, title_font=dict(color=text_color), tickfont=dict(color=text_color)), 
        yaxis=dict(showgrid=True, gridcolor=grid_color, gridwidth=0.1, title_font=dict(color=text_color), tickfont=dict(color=text_color))
    )
    fig.update_traces(line_color='#0068C9')
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: REAL DATA ---
with tab2:
    @st.cache_data
    def load_data():
        dates = pd.date_range(end=datetime.now(), periods=1000, freq="h")
        prices = 40000 + np.cumsum(np.random.randn(1000)) * 100
        return pd.DataFrame({'Timestamp': dates, 'Price': prices})

    real_df = load_data()
    strategy_df = generate_signals(real_df.copy(), fast_window, slow_window)
    
    c1, c2, c3 = st.columns(3)
    curr_price = strategy_df['Price'].iloc[-1]
    signal_status = "BULLISH 🟢" if strategy_df['Signal'].iloc[-1] == 1 else "BEARISH 🔴"
    
    c1.metric("Current Price", f"${curr_price:,.2f}")
    c2.metric("Signal Status", signal_status)
    c3.metric("Active Strategy", f"MA Crossover ({fast_window}/{slow_window})")

    fig_strat = go.Figure()
    fig_strat.add_trace(go.Scatter(x=strategy_df['Timestamp'], y=strategy_df['Price'], name="Price", line=dict(color='gray', width=1, dash='dot')))
    fig_strat.add_trace(go.Scatter(x=strategy_df['Timestamp'], y=strategy_df['Fast_MA'], name="Fast MA", line=dict(color='#00CFBE', width=2)))
    fig_strat.add_trace(go.Scatter(x=strategy_df['Timestamp'], y=strategy_df['Slow_MA'], name="Slow MA", line=dict(color='#FF4B4B', width=2)))
    
    buys = strategy_df[strategy_df['Entry_Exit'] == 2]
    sells = strategy_df[strategy_df['Entry_Exit'] == -2]
    fig_strat.add_trace(go.Scatter(x=buys['Timestamp'], y=buys['Price'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=15, color='#00FF00')))
    fig_strat.add_trace(go.Scatter(x=sells['Timestamp'], y=sells['Price'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=15, color='#FF0000')))

    fig_strat.update_layout(
        title="Technical Analysis", hovermode="x unified", height=600,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=text_color), 
        xaxis=dict(showgrid=False, title_font=dict(color=text_color), tickfont=dict(color=text_color)), 
        yaxis=dict(showgrid=True, gridcolor=grid_color, gridwidth=0.1, title_font=dict(color=text_color), tickfont=dict(color=text_color))
    )
    st.plotly_chart(fig_strat, use_container_width=True)

# --- TAB 3: FORECAST ---
with tab3:
    st.subheader("Stochastic Price Projection (GBM)")
    sim_prices = simulate_gbm(expected_return, volatility, int(time_steps), start_price=real_df['Price'].iloc[-1])
    sim_dates = [datetime.now() + timedelta(days=i) for i in range(len(sim_prices))]
    
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(x=sim_dates, y=sim_prices, name="Projected Path", line=dict(color='#AB63FA')))
    
    fig_mc.update_layout(
        title=f"Projected {time_steps} Day Path", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=text_color), 
        xaxis=dict(showgrid=False, title_font=dict(color=text_color), tickfont=dict(color=text_color)), 
        yaxis=dict(showgrid=True, gridcolor=grid_color, gridwidth=0.1, title_font=dict(color=text_color), tickfont=dict(color=text_color))
    )
    st.plotly_chart(fig_mc, use_container_width=True)
