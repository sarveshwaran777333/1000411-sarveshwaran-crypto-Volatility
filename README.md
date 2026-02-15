# 1000411-sarveshwaran-crypto-Volatility

# 💰 Crypto Volatility Visualizer

Course: Mathematics for AI-II (FA-2)

# Project Overview
This project transforms a static storyboard into a fully interactive Streamlit Dashboard. The tool serves two main purposes:

Simulate Market Swings: Uses mathematical functions (Sine, Cosine, Random Noise) to model price volatility.

Analyze Real Data: Visualizes actual Bitcoin price trends, volatility ranges (High vs. Low), and trading volume.

This application demonstrates how abstract mathematical concepts (amplitude, frequency, drift) map directly to real-world financial behaviors like risk, market speed, and long-term trends.
# Features

# 1. Market Simulator (Math & Code)

User controls allow for dynamic adjustment of mathematical variables to simulate price action:
1. Amplitude ($A$): Simulates Risk/Volatility.
2. Frequency ($f$): Simulates Speed of market swings.
3. Drift ($D$): Simulates Long-term market trends (Bull/Bear market).
4. Pattern Types:
   1.  Sine/Cosine Waves: Represent cyclical market cycles.
   2.  Random Noise: Represents unpredictable market shocks.
The Math Behind the Code:The price P(t) is generated using the formula:
          P(t) = A.sin(f.t) + D.t + N
           Where N is random Gaussian noise.

# 2. Real Bitcoin Data Analysis

Processes historical cryptocurrency data to show:

Close Price: Line chart of price over time.

Volatility Channel: A visualization comparing High vs. Low prices for specific timestamps.

Volume Analysis: Bar chart indicating trading activity.
