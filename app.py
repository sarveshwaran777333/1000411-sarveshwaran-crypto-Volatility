import streamlit as st

import pandas as pd

import numpy as np

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import os

from datetime import datetime, timedelta



# ==========================================

# 1. PAGE CONFIGURATION & STYLING SETTINGS

# ==========================================

st.set_page_config(

    page_title="Crypto Volatility Dashboard",

    page_icon="💰",

    layout="wide",

    initial_sidebar_state="expanded"

)



# Define a consistent color palette for the application

COLORS = {

    "primary": "#2EC4B6",      # Teal for main price

    "fast_ma": "#FF9F1C",      # Bright Orange

    "slow_ma": "#E71D36",      # Bright Red

    "volume": "#8338EC",       # Purple

    "high": "#2CA02C",         # Green

    "low": "#DA70D6",          # Orchid Pink

    "stable": "#3A86FF",       # Blue

    "volatile": "#FF006E",     # Neon Pink

    "background": "rgba(0,0,0,0)",

    "grid": "#E0E0E0"

}



# ==========================================

# 2. DATA PROCESSING & MATH SIMULATION

# ==========================================

def generate_synthetic_data(amp, freq, drift, noise_level, periods=1500):

    """

    Mathematical Logic Engine (From PPT):

    Generates synthetic market data using Trigonometric functions (Sine),

    Linear Drift (Integral trend), and Random Noise (Market shocks).

    """

    # Time vector

    t = np.linspace(0, 50, periods)

    

    # y(t) = A * sin(w*t) + Drift*t + Noise

    base_price = 40000 + amp * np.sin(freq * 0.1 * t) + (drift * 50 * t)

    noise = np.random.normal(0, noise_level, periods)

    sim_price = base_price + noise

    

    dates = pd.date_range(end=datetime.now(), periods=periods, freq="h")

    

    df_sim = pd.DataFrame({

        "Timestamp": dates,

        "Price": sim_price,

        "High": sim_price + np.random.uniform(amp*0.1, amp*0.5, periods),

        "Low": sim_price - np.random.uniform(amp*0.1, amp*0.5, periods),

        "Volume": np.random.randint(1000, 50000, periods)

    })

    return df_sim



@st.cache_data

def load_market_data(use_simulation=False, sim_params=None):

    """

    Loads raw CSV data or falls back to synthetic data based on user toggle.

    Handles Unix timestamps and missing data interpolation.

    """

    if use_simulation and sim_params:

        return generate_synthetic_data(

            sim_params['amp'], 

            sim_params['freq'], 

            sim_params['drift'], 

            sim_params['noise']

        )



    file_path = "crypto_Currency_data.csv"

    if os.path.exists(file_path):

        try:

            df = pd.read_csv(file_path)

            # Standardize naming

            if "Close" in df.columns: 

                df.rename(columns={"Close": "Price"}, inplace=True)

            

            # Standardize timestamps

            if "Timestamp" in df.columns:

                if df["Timestamp"].dtype != 'O':

                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')

                else:

                    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

            

            df.ffill(inplace=True)

            return df.tail(1500).copy()

        except Exception as e:

            st.error(f"Failed to load CSV: {str(e)}")

            return None

    return None



def compute_technical_indicators(df, fast_w, slow_w):

    """

    Applies mathematical formulas to the dataframe:

    1. Simple Moving Averages (Fast & Slow)

    2. Daily Returns (Percentage Change)

    3. Rolling Standard Deviation (Volatility)

    4. Market Regime Classification

    """

    if df is None or df.empty:

        return df

        

    # Moving Averages

    df['Fast_MA'] = df['Price'].rolling(window=fast_w).mean()

    df['Slow_MA'] = df['Price'].rolling(window=slow_w).mean()

    

    # Volatility Math (Standard Deviation of Returns)

    returns = df['Price'].pct_change()

    rolling_volatility = returns.rolling(window=20).std()

    median_volatility = rolling_volatility.median()

    

    # Classify Regimes based on median split

    df['Regime'] = np.where(rolling_volatility > median_volatility, 'Volatile', 'Stable')

    

    return df



# ==========================================

# 3. PLOTLY VISUALIZATION FUNCTIONS

# ==========================================

def create_price_trend_chart(df, f_win, s_win):

    """Creates Chart 1: Price Trend with MAs"""

    fig = go.Figure()

    fig.add_trace(go.Scattergl(

        x=df["Timestamp"], y=df["Price"], 

        name="Close Price", line=dict(color=COLORS["primary"], width=2)

    ))

    fig.add_trace(go.Scattergl(

        x=df["Timestamp"], y=df["Fast_MA"], 

        name=f"{f_win} Fast MA", line=dict(color=COLORS["fast_ma"], width=2)

    ))

    fig.add_trace(go.Scattergl(

        x=df["Timestamp"], y=df["Slow_MA"], 

        name=f"{s_win} Slow MA", line=dict(color=COLORS["slow_ma"], width=2)

    ))

    fig.update_layout(

        height=400, margin=dict(l=10, r=10, t=30, b=10),

        hovermode="x unified", plot_bgcolor=COLORS["background"]

    )

    return fig



def create_volume_chart(df, f_win, s_win):

    """Creates Chart 2: Dual-Axis Trading Volume vs Price MAs"""

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    

    fig.add_trace(go.Bar(

        x=df["Timestamp"], y=df["Volume"], 

        name="Volume", marker_color=COLORS["volume"], opacity=0.5

    ), secondary_y=False)

    

    fig.add_trace(go.Scattergl(

        x=df["Timestamp"], y=df["Fast_MA"], 

        name=f"{f_win} Fast MA", line=dict(color=COLORS["fast_ma"], width=2)

    ), secondary_y=True)

    

    fig.add_trace(go.Scattergl(

        x=df["Timestamp"], y=df["Slow_MA"], 

        name=f"{s_win} Slow MA", line=dict(color=COLORS["slow_ma"], width=2)

    ), secondary_y=True)

    

    fig.update_layout(

        height=400, margin=dict(l=10, r=10, t=30, b=10),

        hovermode="x unified", plot_bgcolor=COLORS["background"]

    )

    fig.update_yaxes(title_text="Volume Activity", secondary_y=False, showgrid=False)

    fig.update_yaxes(title_text="Price Level", secondary_y=True, showgrid=False)

    return fig



def create_high_low_chart(df, f_win, s_win):

    """Creates Chart 3: Day High/Low Range"""

    fig = go.Figure()

    fig.add_trace(go.Scattergl(

        x=df["Timestamp"], y=df["High"], 

        name="Day High", line=dict(color=COLORS["high"], width=1.5)

    ))

    fig.add_trace(go.Scattergl(

        x=df["Timestamp"], y=df["Low"], 

        name="Day Low", line=dict(color=COLORS["low"], width=1.5)

    ))

    fig.add_trace(go.Scattergl(

        x=df["Timestamp"], y=df["Fast_MA"], 

        name=f"{f_win} Fast MA", line=dict(color=COLORS["fast_ma"], width=2, dash='dot')

    ))

    fig.add_trace(go.Scattergl(

        x=df["Timestamp"], y=df["Slow_MA"], 

        name=f"{s_win} Slow MA", line=dict(color=COLORS["slow_ma"], width=2, dash='dot')

    ))

    fig.update_layout(

        height=400, margin=dict(l=10, r=10, t=30, b=10),

        hovermode="x unified", plot_bgcolor=COLORS["background"]

    )

    return fig



def create_regime_chart(df, f_win, s_win):

    """Creates Chart 4: Stability vs Volatility Regimes (Scatter Plot)"""

    fig = go.Figure()

    

    stable_df = df[df['Regime'] == 'Stable']

    volatile_df = df[df['Regime'] == 'Volatile']

    

    fig.add_trace(go.Scattergl(

        x=stable_df["Timestamp"], y=stable_df["Price"], 

        mode='markers', name="Stable Period", 

        marker=dict(color=COLORS["stable"], size=5)

    ))

    fig.add_trace(go.Scattergl(

        x=volatile_df["Timestamp"], y=volatile_df["Price"], 

        mode='markers', name="Volatile Period", 

        marker=dict(color=COLORS["volatile"], size=5)

    ))

    

    fig.add_trace(go.Scattergl(

        x=df["Timestamp"], y=df["Fast_MA"], 

        name=f"{f_win} Fast MA", line=dict(color=COLORS["fast_ma"], width=2)

    ))

    fig.add_trace(go.Scattergl(

        x=df["Timestamp"], y=df["Slow_MA"], 

        name=f"{s_win} Slow MA", line=dict(color=COLORS["slow_ma"], width=2)

    ))

    fig.update_layout(

        height=400, margin=dict(l=10, r=10, t=30, b=10),

        hovermode="x unified", plot_bgcolor=COLORS["background"]

    )

    return fig



# ==========================================

# 4. MAIN APPLICATION ENTRY POINT

# ==========================================

def main():

    # --- SIDEBAR UI ---

    with st.sidebar:

        st.image("https://cdn-icons-png.flaticon.com/512/2875/2875887.png", width=60)

        st.title("🎛️ Analytics Engine")

        st.markdown("Developed by: **Sarveshwaran.K**\n\nCourse: **Mathematics for AI-I**")

        st.markdown("---")

        

        st.subheader("📈 Moving Average Settings")

        fast_window = st.slider("Fast MA Period (Short-term)", 5, 50, 10, help="Reacts quickly to price changes.")

        slow_window = st.slider("Slow MA Period (Long-term)", 20, 200, 50, help="Shows the overarching trend.")

        

        st.markdown("---")

        st.subheader("🌊 Mathematical Simulation")

        use_sim = st.toggle("Enable Math Simulator", value=False, help="Overrides CSV with mathematical waves.")

        

        sim_params = {}

        if use_sim:

            st.caption("Adjust the wave properties:")

            sim_params['amp'] = st.slider("Amplitude (Swing Size)", 100, 5000, 1000)

            sim_params['freq'] = st.slider("Frequency (Speed)", 1, 50, 10)

            sim_params['drift'] = st.slider("Market Drift (Trend)", -5.0, 5.0, 1.0)

            sim_params['noise'] = st.slider("Random Noise (Shocks)", 0, 1000, 200)



    # --- DATA PIPELINE ---

    raw_df = load_market_data(use_sim, sim_params)

    

    if raw_df is None:

        st.error("❌ Fatal Error: Could not load `crypto_Currency_data.csv`.")

        st.info("Please ensure the CSV is uploaded or enable the 'Math Simulator' in the sidebar to generate data.")

        return



    # Process mathematical indicators

    df = compute_technical_indicators(raw_df.copy(), fast_window, slow_window)



    # --- MAIN DASHBOARD UI ---

    st.title("📊 Crypto Volatility & Mathematical Analysis")

    

    # Summary Metrics Row

    st.markdown("### 📌 Real-Time Market Overview")

    metrics_cols = st.columns(4)

    current_price = df.iloc[-1]['Price']

    prev_price = df.iloc[-2]['Price']

    price_change = ((current_price - prev_price) / prev_price) * 100

    

    current_regime = df.iloc[-1]['Regime']

    regime_color = "normal" if current_regime == "Stable" else "inverse"

    

    with metrics_cols[0]:

        st.metric("Latest Price", f"${current_price:,.2f}", f"{price_change:.2f}%")

    with metrics_cols[1]:

        st.metric("Current Regime", current_regime, delta="Volatility Status", delta_color=regime_color)

    with metrics_cols[2]:

        st.metric("Total Data Points", f"{len(df):,}")

    with metrics_cols[3]:

        st.metric("Latest Volume", f"{df.iloc[-1]['Volume']:,.0f}")



    st.markdown("---")



    # Tabs for Organization

    tab_dash, tab_math, tab_data = st.tabs([

        "📈 Interactive Dashboard", 

        "📚 Mathematical Foundations", 

        "🗃️ Raw Dataset"

    ])



    # TAB 1: DASHBOARD (4 CHARTS)

    with tab_dash:

        col1, col2 = st.columns(2)

        

        with col1:

            st.subheader("1. Price Trend Analysis")

            st.plotly_chart(create_price_trend_chart(df, fast_window, slow_window), use_container_width=True)

            

            st.subheader("2. Market Momentum (Volume)")

            st.plotly_chart(create_volume_chart(df, fast_window, slow_window), use_container_width=True)

            

        with col2:

            st.subheader("3. Intra-day Volatility Spread")

            st.plotly_chart(create_high_low_chart(df, fast_window, slow_window), use_container_width=True)

            

            st.subheader("4. Statistical Regime Distribution")

            st.plotly_chart(create_regime_chart(df, fast_window, slow_window), use_container_width=True)



    # TAB 2: MATHEMATICAL FOUNDATIONS (From PPT Presentation)

    with tab_math:

        st.header("🧠 Learner Focused Questions & Mathematics")

        st.markdown("""

        This section explains the mathematical logic powering the dashboard, as presented in the **Mathematics for AI-I** course.

        """)
        with st.expander("1. On Volatility & Spread Analysis", expanded=True):

            st.write("""

            **Question:** If the 'High' and 'Low' prices for a day are very far apart, what does that tell us about the risk level?

            * **Risk Indicator:** A wide gap between High and Low indicates high standard deviation (Volatility).

            * **Mathematical View:** High spreads increase the probability distribution tails, meaning higher risk of sudden loss.

            """)
        with st.expander("2. On Volume and Visual Patterns"):

            st.write("""

            **Question:** What happens to the visual trend if we see a consistent increase in 'Volume' alongside rising 'Close' prices?

            * **Confirmation:** Volume acts as the "weight" or "fuel" in our mathematical vector. Upward momentum coupled with high volume confirms a mathematically strong trend (Bullish).

            """)
        with st.expander("3. Turning Math into Market Movements (The Simulator)"):

            st.write("""

            By enabling the Math Simulator in the sidebar, we use trigonometry and calculus to simulate a market:

            * **Amplitude (`A * sin(wt)`)**: Adjusts the size of the swings. Larger amplitude simulates high-risk assets.

            * **Frequency**: Adjusts the speed. Higher frequency means more waves in less time.

            * **Drift**: Uses integral slopes (`y = mx + b`) to create a long-term upward or downward bias.

            * **Noise**: Injects randomized Gaussian noise to simulate unpredictable market shocks.

            """)
        with st.expander("4. References & Citations"):

            st.markdown("""

            * **Data Processing:** Harris, C.R. et al. (2020). *Array programming with NumPy*. Nature, 585, 357–362.

            * **Volatility Analysis:** Katsiampa, P. (2017). "Volatility estimation for Bitcoin: A comparison of GARCH models." *Economics Letters*.

            * **Technical Indicators:** Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*.

            """)

    with tab_data:

        st.subheader("🗃️ Raw Computed Dataset")

        st.write("View the raw CSV data merged with our newly calculated mathematical indicators.")

        st.dataframe(

            df[["Timestamp", "Price", "High", "Low", "Volume", "Fast_MA", "Slow_MA", "Regime"]].tail(100),

            use_container_width=True,

            height=400
        )

        st.caption("Displaying the last 100 entries for performance.")

if __name__ == "__main__":

    main()
