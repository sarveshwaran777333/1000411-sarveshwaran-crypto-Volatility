# 1000411-sarveshwaran-crypto-Volatility

Crypto Volatility Visualizer: Simulating Market Swings

Bridging the gap between mathematical models and real-world financial behavior.

# Project Overview

Crypto Volatility Visualizer is an interactive web application developed for FinTechLab Pvt. Ltd. to help financial managers and students understand price instability. By combining mathematical simulations with real Bitcoin data, this dashboard allows users to "see" how abstract concepts like amplitude and frequency translate into financial risk and market cycles.

# User Focus

Target Audience: Financial analysts, crypto traders, and AI mathematics students.
Problem Solved: Makes abstract mathematical concepts (Sine waves, Random Noise, Drift) tangible by applying them to financial contexts.
Design Philosophy: "Math for AI" – demonstrating that complex market behaviors can be modeled using fundamental mathematical functions.

# Key Features

As per the FA-2 project brief, this application integrates the following:
Market Simulator: Uses Python code to generate synthetic price patterns (Waves vs. Jumps).
Interactive Controls: Sliders to adjust Amplitude (Risk), Frequency (Speed), and Drift (Trend) in real-time.
Real-Data Analysis: Visualizes actual Bitcoin price history including Close Price and Volume.
Volatility Channel: A specialized graph comparing "High" vs "Low" prices to visualize daily volatility ranges.
Dynamic Dashboard: Built with Streamlit to allow instant updates without refreshing the page.

# Integration & Logic

Core Constructs: The app utilizes NumPy arrays for vectorised mathematical calculations and Pandas DataFrames for handling time-series data.
Mathematical Logic:
   1. Cycles: Modeled using Sine/Cosine functions ($\sin(ft)$).
   2. Trends: Modeled using Linear functions ($y = mx + c$).
   3. Shocks: Modeled using Gaussian Random Noise ($N(\mu, \sigma)$).
UI/UX: Developed in Streamlit with a split-layout design: Sidebar for controls and Main Panel for visualizations, ensuring ease of use for non-technical managers.

# Deployment Instructions

To view the project, you can visit the Web App Link:

[https://1000411-sarveshwaran-crypto-volatility-kkgiunuur5s9vmjbnkkmg8.streamlit.app/]

To run locally:

Clone this repository.
Ensure you have the dependencies installed:
pip install -r requirements.txt
Run the command:
streamlit run app.py

# Application Flow

The Crypto Volatility Visualizer follows a logical flow from parameter setting to visual analysis:

User Input Stage (Control Panel):

Pattern Selection: The user selects a base pattern (Sine Wave, Cosine Wave, or Random Noise).
Parameter Tuning: The user adjusts sliders for Amplitude (Swing Size), Frequency (Speed), and Drift (Slope).
Processing & Logic Stage (The "Math Engine"):
Simulation Calculation: The Python backend computes the price array using the formula:
Price = (Amp x Wave) + (Drift x Time) + Noise

Data Loading: If enabled, the system loads crypto_Currency_data.csv, converts timestamps, and cleans missing values (using dropna).
Visualization Stage (The Dashboard):
Simulation Graph: Displays the mathematically generated price curve using Plotly.
Real Market Comparison: Displays the actual Bitcoin "Close Price" over time for comparison.
Volume & Volatility: Renders bar charts for Volume and channel charts for High/Low data.
Feedback & Insight Stage (Decision Making):
Metric Display: Shows key statistics like "Current Price" and "Highest/Lowest" values.
Visual Analysis: Helps users visually distinguish between "Stable" periods (flat lines) and "Volatile" periods (sharp spikes).

# Repository Structure

app.py: Main Python file containing the Streamlit interface, math logic, and visualization code.

requirements.txt: List of necessary Python packages (streamlit, pandas, plotly, numpy) for cloud deployment.

crypto_Currency_data.csv: The dataset containing real Bitcoin delivery data (Timestamp, Open, High, Low, Close, Volume).

# Storyboard & Screenshots

Storyboard: [https://www.canva.com/design/DAHAVel0ek8/3eR8BJragBl17uuymFyoDQ/view?utm_content=DAHAVel0ek8&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h9a7612c023]

# Screenshots:

<img width="529" height="536" alt="Screenshot 2026-02-15 192708" src="https://github.com/user-attachments/assets/deb5d86b-d920-4e9e-b7aa-f78f9447638a" />

<img width="1497" height="696" alt="Screenshot 2026-02-15 192734" src="https://github.com/user-attachments/assets/91134680-3a19-4173-8acb-e3230d5878ae" />

<img width="1475" height="671" alt="Screenshot 2026-02-15 192913" src="https://github.com/user-attachments/assets/3552def4-1438-4701-9ec7-741a1a113466" />

<img width="1473" height="555" alt="Screenshot 2026-02-15 192953" src="https://github.com/user-attachments/assets/ee5e5b38-f798-4169-9f3e-38ae0164bd27" />

<img width="1455" height="597" alt="Screenshot 2026-02-15 193027" src="https://github.com/user-attachments/assets/27796f72-c1a2-4c0c-a162-634edd7aa525" />

# Tested By

Saif: Tested the slider interactivity and graph responsiveness.

Sister: tested design and logic part of the app.

# Credits & Acknowledgements

This project was developed as part of the Formative Assessment-2 (FA-2) for the Mathematics for AI-II course.

Course: Mathematics for AI-II

# Assignment: Crypto Volatility Visualizer

Student Name: Sarveshwaran.K

Student ID: 1000411

Context: FinTechLab Pvt. Ltd.
Volume Analysis: Bar chart indicating trading activity.
