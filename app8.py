import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Type-0 SPDC Simulation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 0.5rem 0;
    }
    .physics-card {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header"><h1>Type-0 SPDC Simulation</h1><p>Spontaneous Parametric Down-Conversion in PPKTP Crystal</p></div>', unsafe_allow_html=True)

# Sellmeier equation for PPKTP
@st.cache_data
def sellmeier_ppktp(wavelength_um, temperature_c):
    """
    Sellmeier equation for PPKTP (KTiOPO4)
    Based on Kato and Takaoka (1986) with temperature dependence
    """
    T = temperature_c + 273.15
    T0 = 298.15  # Reference temperature (25°C)

    # Sellmeier coefficients for PPKTP
    # Extraordinary ray (e-ray) for Type-0
    A1_e = 3.0333
    A2_e = 0.04154
    A3_e = 0.04547
    A4_e = -0.01408
    A5_e = -0.01977
    A6_e = -0.01204

    # Temperature coefficients
    dn_dT_e = -1.6e-5  # /K

    lambda_sq = wavelength_um ** 2

    # Extraordinary refractive index
    n_e_sq = A1_e + A2_e / (lambda_sq - A3_e) + A4_e / (lambda_sq - A5_e) + A6_e * lambda_sq
    n_e = np.sqrt(n_e_sq)

    # Apply temperature correction
    n_e = n_e + dn_dT_e * (T - T0)

    return n_e

@st.cache_data
def calculate_phase_mismatch_type0(pump_wl, signal_wl, idler_wl, temperature, poling_period):
    """
    Calculate phase mismatch for Type-0 SPDC process
    All waves have the same polarization (extraordinary)
    """
    # Convert wavelengths to micrometers
    pump_um = pump_wl / 1000
    signal_um = signal_wl / 1000
    idler_um = idler_wl / 1000

    # Get refractive indices (all extraordinary for Type-0)
    n_pump = sellmeier_ppktp(pump_um, temperature)
    n_signal = sellmeier_ppktp(signal_um, temperature)
    n_idler = sellmeier_ppktp(idler_um, temperature)

    # Wave vectors
    k_pump = 2 * np.pi * n_pump / pump_um
    k_signal = 2 * np.pi * n_signal / signal_um
    k_idler = 2 * np.pi * n_idler / idler_um

    # Quasi-phase matching wave vector
    k_qpm = 2 * np.pi / poling_period

    # Phase mismatch
    delta_k = k_pump - k_signal - k_idler - k_qpm

    return delta_k

def gaussian(x, amplitude, center, width, offset):
    return amplitude * np.exp(-((x - center) / width) ** 2) + offset

def calculate_spdc_spectrum(wavelength, pump_wavelength, crystal_length, temperature, poling_period, 
                          spectral_width, asymmetry_strength, noise_level):
    """Calculate the SPDC spectrum with realistic effects"""
    
    center_wavelength = 2 * pump_wavelength  # Degenerate wavelength
    
    # Calculate idler wavelengths using energy conservation
    idler_wavelength = 1 / (1/pump_wavelength - 1/wavelength)
    
    # Calculate phase mismatch for each wavelength pair
    delta_k_array = []
    for i, sig_wl in enumerate(wavelength):
        if idler_wavelength[i] > 0:  # Valid idler wavelength
            dk = calculate_phase_mismatch_type0(pump_wavelength, sig_wl, idler_wavelength[i], 
                                             temperature, poling_period)
            delta_k_array.append(dk)
        else:
            delta_k_array.append(np.inf)
    
    delta_k_array = np.array(delta_k_array)
    
    # Calculate phase matching efficiency using sinc function
    L_um = crystal_length * 1000  # Convert to micrometers
    phase_efficiency = np.sinc(delta_k_array * L_um / (2 * np.pi)) ** 2
    
    # Handle infinite values
    phase_efficiency[np.isinf(delta_k_array)] = 0
    phase_efficiency[np.isnan(phase_efficiency)] = 0
    
    # Create base Gaussian curve
    amplitude = 1.0
    offset = 0.05
    
    # Generate the base spectrum
    base_spectrum = gaussian(wavelength, amplitude, center_wavelength, spectral_width, offset)
    
    # Multiply by phase matching efficiency
    spectrum = base_spectrum * phase_efficiency
    
    # Add asymmetry
    asymmetry = asymmetry_strength * np.exp(-((wavelength - center_wavelength - 2) / 15.0) ** 2)
    spectrum += asymmetry
    
    # Add baseline variation
    baseline_variation = 0.02 * np.sin((wavelength - wavelength[0]) * 0.08) * np.cos((wavelength - wavelength[0]) * 0.03)
    spectrum += baseline_variation
    
    # Add noise
    spectrum += np.random.normal(0, noise_level, len(spectrum))
    
    # Ensure non-negative values
    spectrum = np.maximum(spectrum, 0)
    
    # Apply light smoothing
    spectrum = uniform_filter1d(spectrum, size=3)
    
    # Normalize spectrum
    if np.max(spectrum) > 0:
        spectrum = spectrum / np.max(spectrum)
    
    return spectrum, phase_efficiency, delta_k_array

# Sidebar controls
st.sidebar.header("Crystal Parameters")

# Basic parameters
pump_wavelength = st.sidebar.slider("Pump Wavelength (nm)", 350, 500, 405, 1)
crystal_length = st.sidebar.slider("Crystal Length (mm)", 5, 100, 30, 1)
temperature = st.sidebar.slider("Temperature (°C)", 0, 100, 25, 1)
poling_period = st.sidebar.slider("Poling Period (μm)", 1.0, 10.0, 3.425, 0.1)

st.sidebar.header("Spectral Properties")

# Spectral range
min_wavelength = st.sidebar.slider("Min Wavelength (nm)", 700, 800, 780, 1)
max_wavelength = st.sidebar.slider("Max Wavelength (nm)", 840, 900, 840, 1)

# Spectrum shaping parameters
spectral_width = st.sidebar.slider("Spectral Width", 8.0, 20.0, 12.0, 0.5)
asymmetry_strength = st.sidebar.slider("Asymmetry Strength", 0.0, 0.2, 0.05, 0.01)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.005, 0.0008, 0.0001)

st.sidebar.header("Analysis Options")
show_refractive_indices = st.sidebar.checkbox("Show Refractive Indices", value=True)

# Generate wavelength array
wavelength = np.linspace(min_wavelength, max_wavelength, 1000)

# Calculate spectrum
spectrum, phase_efficiency, delta_k_array = calculate_spdc_spectrum(
    wavelength, pump_wavelength, crystal_length, temperature, poling_period,
    spectral_width, asymmetry_strength, noise_level
)

# Main plot
st.header("Type-0 SPDC Spectrum")

fig, ax = plt.subplots(figsize=(12, 8))

# Plot spectrum
ax.plot(wavelength, spectrum, 'red', linewidth=2.5, label='Type-0 SPDC Spectrum')
ax.fill_between(wavelength, spectrum, alpha=0.3, color='red')

# Calculate and show FWHM
half_max = np.max(spectrum) / 2
indices = np.where(spectrum >= half_max)[0]
if len(indices) > 0:
    fwhm_indices = [indices[0], indices[-1]]
    fwhm_bandwidth = wavelength[fwhm_indices[-1]] - wavelength[fwhm_indices[0]]
    
    # Add FWHM markers
    ax.axhline(y=half_max, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=wavelength[fwhm_indices[0]], color='blue', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=wavelength[fwhm_indices[-1]], color='blue', linestyle='--', alpha=0.7, linewidth=1)
    ax.text(wavelength[fwhm_indices[0]] + 2, half_max + 0.1, f'FWHM = {fwhm_bandwidth:.1f} nm',
            fontsize=12, color='blue', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))

# Add peak wavelength marker
peak_idx = np.argmax(spectrum)
peak_wavelength_actual = wavelength[peak_idx]
ax.axvline(x=peak_wavelength_actual, color='green', linestyle=':', alpha=0.7, linewidth=2)
ax.text(peak_wavelength_actual + 1, 0.9, f'Peak: {peak_wavelength_actual:.1f} nm',
        fontsize=12, color='green', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))

# Add degenerate wavelength marker
degenerate_wl = pump_wavelength * 2
if min_wavelength <= degenerate_wl <= max_wavelength:
    ax.axvline(x=degenerate_wl, color='purple', linestyle='-.', alpha=0.7, linewidth=2)
    ax.text(degenerate_wl + 1, 0.8, f'Degenerate: {degenerate_wl} nm',
            fontsize=12, color='purple', bbox=dict(boxstyle="round,pad=0.3", facecolor="plum", alpha=0.5))

ax.set_xlabel('Wavelength (nm)', fontsize=14)
ax.set_ylabel('Normalized Intensity', fontsize=14)
ax.set_title('Type-0 SPDC Spectrum - PPKTP Crystal', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_xlim(min_wavelength, max_wavelength)
ax.set_ylim(0, 1.1)

plt.tight_layout()
st.pyplot(fig)

# Analysis columns
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Spectral Analysis</h4>
        <p><strong>Peak Wavelength:</strong> {peak_wavelength_actual:.1f} nm</p>
        <p><strong>Degenerate Wavelength:</strong> {degenerate_wl} nm</p>
        <p><strong>Peak Intensity:</strong> {np.max(spectrum):.3f}</p>
        <p><strong>FWHM Bandwidth:</strong> {fwhm_bandwidth:.1f} nm</p>
        <p><strong>Max Phase Efficiency:</strong> {np.max(phase_efficiency):.3f}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="physics-card">
        <h4>Crystal Properties</h4>
        <p><strong>Crystal Length:</strong> {crystal_length} mm</p>
        <p><strong>Poling Period:</strong> {poling_period} μm</p>
        <p><strong>Temperature:</strong> {temperature} °C</p>
        <p><strong>Pump Wavelength:</strong> {pump_wavelength} nm</p>
        <p><strong>Configuration:</strong> Type-0 SPDC</p>
    </div>
    """, unsafe_allow_html=True)

# Refractive indices
if show_refractive_indices:
    st.header("Refractive Indices")
    
    # Calculate refractive indices for key wavelengths
    key_wavelengths = [pump_wavelength, peak_wavelength_actual, degenerate_wl]
    refractive_data = []
    
    for wl in key_wavelengths:
        if min_wavelength <= wl <= max_wavelength:
            n_e = sellmeier_ppktp(wl/1000, temperature)
            refractive_data.append({
                'Wavelength (nm)': wl,
                'Refractive Index (n_e)': f"{n_e:.4f}",
                'Type': 'Extraordinary'
            })
    
    if refractive_data:
        df_refractive = pd.DataFrame(refractive_data)
        st.dataframe(df_refractive, use_container_width=True)
    
    # Plot refractive index vs wavelength
    wl_range = np.linspace(min_wavelength, max_wavelength, 200)
    n_values = [sellmeier_ppktp(wl/1000, temperature) for wl in wl_range]
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(wl_range, n_values, 'purple', linewidth=2, label=f'n_e at {temperature}°C')
    ax3.set_xlabel('Wavelength (nm)', fontsize=12)
    ax3.set_ylabel('Refractive Index', fontsize=12)
    ax3.set_title('Refractive Index vs Wavelength for PPKTP', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    st.pyplot(fig3)

# Temperature tuning analysis


temp_range = np.linspace(max(0, temperature-20), min(100, temperature+20), 50)
peak_shifts = []
max_efficiencies = []

for temp in temp_range:
    # Calculate spectrum for this temperature
    spectrum_temp, phase_eff_temp, _ = calculate_spdc_spectrum(
        wavelength, pump_wavelength, crystal_length, temp, poling_period,
        spectral_width, asymmetry_strength, 0  # No noise for smooth curves
    )
    
    # Find peak
    peak_idx_temp = np.argmax(spectrum_temp)
    peak_shifts.append(wavelength[peak_idx_temp])
    max_efficiencies.append(np.max(phase_eff_temp))

fig4, (ax4, ax5) = plt.subplots(2, 1, figsize=(12, 10))

# Peak wavelength vs temperature
ax4.plot(temp_range, peak_shifts, 'r-', linewidth=2, marker='o', markersize=4)
ax4.axhline(y=degenerate_wl, color='purple', linestyle='--', alpha=0.7, linewidth=1, label=f'Degenerate ({degenerate_wl} nm)')
ax4.axvline(x=temperature, color='green', linestyle=':', alpha=0.7, linewidth=2, label=f'Current T ({temperature}°C)')
ax4.set_xlabel('Temperature (°C)', fontsize=12)
ax4.set_ylabel('Peak Wavelength (nm)', fontsize=12)
ax4.set_title('Peak Wavelength vs Temperature', fontsize=14)
ax4.grid(True, alpha=0.3)
ax4.legend()

# Max efficiency vs temperature
ax5.plot(temp_range, max_efficiencies, 'b-', linewidth=2, marker='s', markersize=4)
ax5.axvline(x=temperature, color='green', linestyle=':', alpha=0.7, linewidth=2, label=f'Current T ({temperature}°C)')
ax5.set_xlabel('Temperature (°C)', fontsize=12)
ax5.set_ylabel('Maximum Efficiency', fontsize=12)
ax5.set_title('Maximum Phase Matching Efficiency vs Temperature', fontsize=14)
ax5.grid(True, alpha=0.3)
ax5.legend()

plt.tight_layout()
st.pyplot(fig4)

# Export data
# st.header("💾 Export Data")

# # Create dataframe for export
# export_data = pd.DataFrame({
#     'Wavelength (nm)': wavelength,
#     'Normalized Intensity': spectrum,
#     'Phase Efficiency': phase_efficiency,
#     'Phase Mismatch (1/μm)': delta_k_array
# })

# # Remove infinite values for export
# export_data = export_data.replace([np.inf, -np.inf], np.nan)

# col5, col6 = st.columns(2)

# with col5:
#     st.download_button(
#         label="📥 Download Spectrum Data (CSV)",
#         data=export_data.to_csv(index=False),
#         file_name=f"type0_spdc_spectrum_T{temperature}C.csv",
#         mime="text/csv"
#     )

# with col6:
#     # Summary statistics
#     summary_stats = {
#         'Parameter': ['Pump Wavelength', 'Peak Wavelength', 'FWHM', 'Crystal Length', 'Temperature', 'Poling Period', 'Max Efficiency'],
#         'Value': [f"{pump_wavelength} nm", f"{peak_wavelength_actual:.1f} nm", f"{fwhm_bandwidth:.1f} nm", 
#                  f"{crystal_length} mm", f"{temperature} °C", f"{poling_period} μm", f"{np.max(phase_efficiency):.3f}"]
#     }
    
#     summary_df = pd.DataFrame(summary_stats)
#     st.download_button(
#         label="📊 Download Summary (CSV)",
#         data=summary_df.to_csv(index=False),
#         file_name=f"type0_spdc_summary_T{temperature}C.csv",
#         mime="text/csv"
#     )

# Footer with information
st.markdown("---")
st.markdown("""
**About Type-0 SPDC:**
- Type-0 SPDC: All three waves (pump, signal, idler) have the same polarization
- Phase matching achieved through quasi-phase matching (QPM) in periodically poled crystals
- Temperature tuning provides fine control over phase matching conditions
- PPKTP (KTiOPO₄) is commonly used for near-IR applications
""")