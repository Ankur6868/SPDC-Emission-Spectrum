import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set page config
st.set_page_config(page_title="PPKTP SPDC Simulator", layout="wide")

# Title and description
st.title("PPKTP SPDC Poling Period Simulator")
st.markdown("""
This simulator calculates the poling period for Type-0 spontaneous parametric down-conversion (SPDC) 
in periodically poled potassium titanyl phosphate (PPKTP) crystals.
""")

# Sellmeier + thermo-optic coefficients for PPKTP (Y-polarization)
Ay, By, Cy, Dy = 2.09930, 0.922683, 0.0467695, 0.0138404

# Functions for degenerate case
def nY_base(lam_um):
    return np.sqrt(Ay + By / (1 - Cy / lam_um**2) - Dy * lam_um**2)

def dnY_dT_degenerate(lam_um):
    n1Y = [6.2897, 6.3061, -6.0629, 2.6486]
    return sum(a / lam_um**m for m, a in enumerate(n1Y)) * 1e-6

def d2nY_dT2_degenerate(lam_um):
    n2Y = [-0.14445, 2.2244, -3.5770, 1.3470]
    return sum(a / lam_um**m for m, a in enumerate(n2Y)) * 1e-8

def nY_temp_degenerate(lam_um, T_C):
    ΔT = T_C - 25.0
    return nY_base(lam_um) + dnY_dT_degenerate(lam_um) * ΔT + d2nY_dT2_degenerate(lam_um) * ΔT**2

def poling_period_vs_lambda_degenerate(lam_um, T_C):
    lam_p = lam_um / 2
    np_p = nY_temp_degenerate(lam_p, T_C)
    ns = nY_temp_degenerate(lam_um, T_C)
    return 1 / ((np_p / lam_p - 2 * ns / lam_um))

# Functions for non-degenerate case
def dn_y_dT_nondegenerate(λ):
    return (1.997 * λ**3 - 4.067 * λ**2 + 5.154 * λ - 5.425) * 1e-6

def n_y_nondegenerate(λ, T):
    n_squared = Ay + By / (1 - Cy / λ**2) - Dy * λ**2
    return np.sqrt(n_squared) + dn_y_dT_nondegenerate(λ) * (T - 25)

def lambda_i(lambda_p, lambda_s):
    return 1.0 / (1.0 / lambda_p - 1.0 / lambda_s)

def poling_period_nondegenerate(lambda_s, lambda_p, T):
    lambda_i_val = lambda_i(lambda_p, lambda_s)
    n_p = n_y_nondegenerate(lambda_p, T)
    n_s = n_y_nondegenerate(lambda_s, T)
    n_i = n_y_nondegenerate(lambda_i_val, T)
    return 1.0 / (n_p / lambda_p - n_s / lambda_s - n_i / lambda_i_val)

# Sidebar controls
st.sidebar.header("Simulation Parameters")

# Temperature range
temp_min = st.sidebar.slider("Minimum Temperature (°C)", 20, 80, 25)
temp_max = st.sidebar.slider("Maximum Temperature (°C)", 30, 150, 100)
temp_step = st.sidebar.slider("Temperature Step (°C)", 5, 25, 25)

temperatures = np.arange(temp_min, temp_max + temp_step, temp_step)

# Main content with tabs
tab1, tab2, tab3 = st.tabs(["Degenerate SPDC", "Non-Degenerate SPDC", "Comparison"])

with tab1:
    st.header("Degenerate SPDC")
    st.markdown("In degenerate SPDC, the pump photon splits into two identical photons (signal and idler).")
    
    # Wavelength range for degenerate
    col1, col2 = st.columns(2)
    with col1:
        lambda_min_deg = st.slider("Min Wavelength (nm)", 700, 900, 750, key="deg_min")
        lambda_max_deg = st.slider("Max Wavelength (nm)", 1000, 1300, 1200, key="deg_max")
    with col2:
        num_points_deg = st.slider("Number of Points", 100, 1000, 500, key="deg_points")
        show_grid_deg = st.checkbox("Show Grid", True, key="deg_grid")
    
    λ_um_deg = np.linspace(lambda_min_deg/1000, lambda_max_deg/1000, num_points_deg)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    for T in temperatures:
        try:
            Λ_um_deg = poling_period_vs_lambda_degenerate(λ_um_deg, T)
            # Filter out unrealistic values
            valid_mask = (Λ_um_deg > 0) & (Λ_um_deg < 1000)
            if np.any(valid_mask):
                ax1.plot(λ_um_deg[valid_mask] * 1e3, Λ_um_deg[valid_mask], 
                        label=f"T = {T}°C", linewidth=2)
        except:
            continue
    
    ax1.set_xlabel("Signal/Idler Wavelength λ (nm)")
    ax1.set_ylabel("Poling Period Λ (µm)")
    ax1.set_title("Poling Period vs Wavelength for Degenerate SPDC in PPKTP")
    if show_grid_deg:
        ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    
    st.pyplot(fig1)

with tab2:
    st.header("Non-Degenerate SPDC")
    st.markdown("In non-degenerate SPDC, the pump photon splits into two different photons (signal and idler).")
    
    # Parameters for non-degenerate
    col1, col2 = st.columns(2)
    with col1:
        lambda_p_nm = st.slider("Pump Wavelength (nm)", 350, 500, 405, key="pump_wl")
        lambda_s_min = st.slider("Min Signal Wavelength (nm)", 600, 800, 700, key="sig_min")
        lambda_s_max = st.slider("Max Signal Wavelength (nm)", 800, 1000, 900, key="sig_max")
    with col2:
        num_points_nondeg = st.slider("Number of Points", 100, 500, 300, key="nondeg_points")
        show_grid_nondeg = st.checkbox("Show Grid", True, key="nondeg_grid")
    
    lambda_p = lambda_p_nm / 1000  # Convert to µm
    lambda_s_range = np.linspace(lambda_s_min/1000, lambda_s_max/1000, num_points_nondeg)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for T in temperatures:
        try:
            Lambda_vals = []
            valid_lambdas = []
            
            for ls in lambda_s_range:
                try:
                    # Check if idler wavelength is valid
                    li = lambda_i(lambda_p, ls)
                    if li > 0 and li < 5:  # Reasonable range for idler
                        Lambda_val = poling_period_nondegenerate(ls, lambda_p, T)
                        if Lambda_val > 0 and Lambda_val < 1000:  # Reasonable range for poling period
                            Lambda_vals.append(Lambda_val * 1e6)  # Convert to µm
                            valid_lambdas.append(ls)
                except:
                    continue
            
            if Lambda_vals:
                ax2.plot(np.array(valid_lambdas) * 1000, Lambda_vals, 
                        label=f"T = {T}°C", linewidth=2)
        except:
            continue
    
    ax2.set_xlabel("Signal Wavelength λₛ (nm)")
    ax2.set_ylabel("Poling Period Λ (µm)")
    ax2.set_title(f"Poling Period vs Signal Wavelength for Non-Degenerate SPDC (λₚ = {lambda_p_nm} nm)")
    if show_grid_nondeg:
        ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    
    st.pyplot(fig2)
    
    # Show idler wavelength information
    st.subheader("Idler Wavelength Information")
    sample_signal = (lambda_s_min + lambda_s_max) / 2
    sample_idler = lambda_i(lambda_p, sample_signal/1000) * 1000
    st.write(f"For signal wavelength {sample_signal:.0f} nm, idler wavelength is {sample_idler:.0f} nm")

with tab3:
    st.header("Comparison")
    st.markdown("Compare the poling periods for both degenerate and non-degenerate cases.")
    
    # Use a fixed temperature for comparison
    comp_temp = st.slider("Comparison Temperature (°C)", int(temp_min), int(temp_max), 50)
    
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Degenerate plot
    λ_um_comp = np.linspace(0.75, 1.2, 500)
    try:
        Λ_um_comp = poling_period_vs_lambda_degenerate(λ_um_comp, comp_temp)
        valid_mask = (Λ_um_comp > 0) & (Λ_um_comp < 1000)
        ax3a.plot(λ_um_comp[valid_mask] * 1e3, Λ_um_comp[valid_mask], 
                 'b-', linewidth=2, label=f"Degenerate (T = {comp_temp}°C)")
    except:
        pass
    
    ax3a.set_xlabel("Signal/Idler Wavelength λ (nm)")
    ax3a.set_ylabel("Poling Period Λ (µm)")
    ax3a.set_title("Degenerate SPDC")
    ax3a.grid(True, alpha=0.3)
    ax3a.legend()
    
    # Non-degenerate plot
    lambda_s_comp = np.linspace(0.7, 0.9, 300)
    try:
        Lambda_comp = []
        valid_lambdas_comp = []
        
        for ls in lambda_s_comp:
            try:
                li = lambda_i(lambda_p, ls)
                if li > 0 and li < 5:
                    Lambda_val = poling_period_nondegenerate(ls, lambda_p, comp_temp)
                    if Lambda_val > 0 and Lambda_val < 1000:
                        Lambda_comp.append(Lambda_val * 1e6)
                        valid_lambdas_comp.append(ls)
            except:
                continue
        
        if Lambda_comp:
            ax3b.plot(np.array(valid_lambdas_comp) * 1000, Lambda_comp, 
                     'r-', linewidth=2, label=f"Non-degenerate (T = {comp_temp}°C)")
    except:
        pass
    
    ax3b.set_xlabel("Signal Wavelength λₛ (nm)")
    ax3b.set_ylabel("Poling Period Λ (µm)")
    ax3b.set_title(f"Non-Degenerate SPDC (λₚ = {lambda_p_nm} nm)")
    ax3b.grid(True, alpha=0.3)
    ax3b.legend()
    
    plt.tight_layout()
    st.pyplot(fig3)

# Information section
st.sidebar.header("Information")
st.sidebar.info("""
**PPKTP Properties:**
- Material: Periodically Poled KTiOPO₄
- Type: Type-0 phase matching
- Polarization: Y-polarization

**Degenerate SPDC:**
- λₛ = λᵢ = 2λₚ
- Both photons have same wavelength

**Non-Degenerate SPDC:**
- λₛ ≠ λᵢ
- Energy conservation: 1/λₚ = 1/λₛ + 1/λᵢ
""")

