# ... [All imports and initial setup remain unchanged] ...
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Pump Beam Coincidence Rate",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> Pump Beam Coincidence Rate Analyzer</h1>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("Model Parameters")
st.sidebar.markdown("---")

# Model parameters
st.sidebar.subheader("Bell Curve Parameters")
A = st.sidebar.slider(
    "Amplitude (A)", 
    min_value=0.1, 
    max_value=5.0, 
    value=1.0, 
    step=0.1,
    help="Controls the overall amplitude of the coincidence rate"
)

B = st.sidebar.slider(
    "Shape Parameter (B)", 
    min_value=500, 
    max_value=3000, 
    value=1400, 
    step=50,
    help="Controls the width and position of the peak"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Range Settings")

# Range parameters
w0_min = st.sidebar.slider(
    "Min Beam Waist (µm)", 
    min_value=10, 
    max_value=40, 
    value=20, 
    step=1
)

w0_max = st.sidebar.slider(
    "Max Beam Waist (µm)", 
    min_value=50, 
    max_value=100, 
    value=60, 
    step=1
)

num_points = st.sidebar.slider(
    "Number of Data Points", 
    min_value=100, 
    max_value=1000, 
    value=300, 
    step=50
)

# Mathematical model display
st.sidebar.markdown("---")
st.sidebar.subheader("Mathematical Model")
st.sidebar.latex(r"Rate = \frac{A}{w_0^2} \exp\left(-\frac{B}{w_0^2}\right)")
st.sidebar.markdown(f"**Current Parameters:**")
st.sidebar.markdown(f"- A = {A}")
st.sidebar.markdown(f"- B = {B}")

# Define the coincidence rate function
def coincidence_rate(w0, A=1.0, B=1400):
    """Calculate coincidence rate based on pump beam waist"""
    return A / (w0**2) * np.exp(-B / w0**2)

# Generate data
w0_vals = np.linspace(w0_min, w0_max, num_points)
rates = coincidence_rate(w0_vals, A, B)

# Find maximum and its position
max_rate = np.max(rates)
max_position = w0_vals[np.argmax(rates)]

# Normalize for plotting
rates_normalized = 100 * rates / max_rate
# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Coincidence Rate vs Pump Beam Size")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(w0_vals, rates_normalized, 'b-', linewidth=3, label="Coincidence Rate", alpha=0.8)
    ax.fill_between(w0_vals, rates_normalized, alpha=0.3, color='lightblue')
    ax.axvline(max_position, linestyle='--', color='red', alpha=0.8, linewidth=2,
               label=f"Max at {max_position:.1f} µm")
    ax.axhline(50, linestyle=':', color='gray', alpha=0.6, label="50% level")

    ax.set_xlabel("Pump beam waist (µm)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Coincidence rate (normalized %)", fontsize=14, fontweight='bold')
    ax.set_title("Coincidence Rate vs Pump Beam Size", fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(w0_min, w0_max)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Key Metrics")
    half_max = max_rate / 2
    indices = np.where(rates >= half_max)[0]
    fwhm = w0_vals[indices[-1]] - w0_vals[indices[0]] if len(indices) > 0 else 0

    st.metric("Maximum Rate", f"{max_rate:.6f}")
    st.metric("Optimal Beam Waist", f"{max_position:.2f} µm")
    st.metric("FWHM", f"{fwhm:.2f} µm")

# Analysis with only Parameter Sensitivity tab
st.markdown("---")
st.subheader("Analysis")

tab1, = st.tabs(["Parameter Sensitivity"])

with tab1:
    st.write("**Effect of Parameter Changes:**")

    col_sens1, col_sens2 = st.columns(2)

    with col_sens1:
        st.write("*Amplitude (A) Sensitivity:*")
        fig_A, ax_A = plt.subplots(figsize=(6, 4))

        A_values = [0.5, 1.0, 1.5, 2.0]
        colors = ['blue', 'green', 'orange', 'red']

        for i, A_test in enumerate(A_values):
            rates_test = coincidence_rate(w0_vals, A_test, B)
            rates_test_norm = 100 * rates_test / np.max(rates_test)
            ax_A.plot(w0_vals, rates_test_norm, color=colors[i],
                      label=f"A = {A_test}", linewidth=2)

        ax_A.set_xlabel("Pump beam waist (µm)")
        ax_A.set_ylabel("Normalized rate (%)")
        ax_A.legend()
        ax_A.grid(True, alpha=0.3)
        ax_A.set_title("Effect of Amplitude (A)")
        plt.tight_layout()
        st.pyplot(fig_A)

    with col_sens2:
        st.write("*Shape Parameter (B) Sensitivity:*")
        fig_B, ax_B = plt.subplots(figsize=(6, 4))

        B_values = [1000, 1400, 1800, 2200]

        for i, B_test in enumerate(B_values):
            rates_test = coincidence_rate(w0_vals, A, B_test)
            rates_test_norm = 100 * rates_test / np.max(rates_test)
            ax_B.plot(w0_vals, rates_test_norm, color=colors[i],
                      label=f"B = {B_test}", linewidth=2)

        ax_B.set_xlabel("Pump beam waist (µm)")
        ax_B.set_ylabel("Normalized rate (%)")
        ax_B.legend()
        ax_B.grid(True, alpha=0.3)
        ax_B.set_title("Effect of Shape Parameter (B)")
        plt.tight_layout()
        st.pyplot(fig_B)

# Footer
st.markdown("---")
st.markdown("""
**About this model:**
- This simulation models the coincidence rate in quantum optics experiments
- The bell-shaped curve represents the optimization of pump beam waist for maximum coincidence rate
- Parameter B controls the position and width of the peak
- Parameter A controls the overall amplitude
- The model assumes ideal conditions and may not account for all experimental factors
""")

# Sidebar result summary
st.sidebar.markdown("---")
st.sidebar.subheader("Current Results")
st.sidebar.metric("Max Rate", f"{max_rate:.6f}")
st.sidebar.metric("Optimal Waist", f"{max_position:.1f} µm")
st.sidebar.metric("FWHM", f"{fwhm:.1f} µm")
