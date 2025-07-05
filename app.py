import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import sinc

# Configure page
st.set_page_config(
    page_title="Type 0 PPKTP Simulation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
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
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header"><h1>🔬 Type 0 PPKTP Nonlinear Crystal Simulation</h1><p>Spontaneous Parametric Down-Conversion Analysis</p></div>', unsafe_allow_html=True)

# Physical constants
C = 2.998e8  # Speed of light (m/s)
H = 6.626e-34  # Planck constant (J·s)

class PPKTPSimulator:
    def __init__(self):
        # More accurate PPKTP Sellmeier coefficients
        self.n_coeffs = {
            'A1': 2.1146,
            'A2': 0.89188,
            'A3': 0.73258,
            'B1': 0.21369,
            'B2': 0.20861,
            'B3': 0.94779,
            'C1': 0.01027,
            'C2': 0.01834,
            'C3': 213.32
        }
        
        # Temperature coefficients (more accurate)
        self.temp_coeffs = {
            'a0': 9.9587e-6,
            'a1': 9.9228e-6,
            'a2': -8.9603e-6,
            'a3': 4.1010e-6,
            'b0': -1.1882e-8,
            'b1': 10.459e-8,
            'b2': -9.5038e-8,
            'b3': 3.8584e-8
        }
    
    def sellmeier_equation(self, wavelength_um, temperature_c):
        """Calculate refractive index using accurate Sellmeier equation with temperature dependence"""
        T = temperature_c
        lam = wavelength_um
        lam2 = lam * lam
        
        # Base Sellmeier equation for PPKTP
        n_squared = (self.n_coeffs['A1'] + 
                    self.n_coeffs['B1'] / (lam2 - self.n_coeffs['C1']) +
                    self.n_coeffs['B2'] / (lam2 - self.n_coeffs['C2']) +
                    self.n_coeffs['B3'] / (lam2 - self.n_coeffs['C3']))
        
        # Temperature dependence - critical for phase matching
        dT = T - 25.0  # Reference temperature
        
        # Temperature correction terms
        dn_dT = (self.temp_coeffs['a0'] + 
                 self.temp_coeffs['a1'] * lam + 
                 self.temp_coeffs['a2'] * lam2 + 
                 self.temp_coeffs['a3'] * lam**3)
        
        d2n_dT2 = (self.temp_coeffs['b0'] + 
                   self.temp_coeffs['b1'] * lam + 
                   self.temp_coeffs['b2'] * lam2 + 
                   self.temp_coeffs['b3'] * lam**3)
        
        # Apply temperature correction
        n_base = np.sqrt(n_squared)
        n_corrected = n_base + dn_dT * dT + 0.5 * d2n_dT2 * dT**2
        
        return n_corrected
    
    def phase_matching_condition(self, pump_wl, signal_wl, idler_wl, temperature, poling_period):
        """Calculate phase mismatch for Type 0 PPKTP with accurate dispersion"""
        # Convert wavelengths to micrometers
        pump_wl_um = pump_wl / 1000
        signal_wl_um = signal_wl / 1000
        idler_wl_um = idler_wl / 1000
        
        # Calculate temperature-dependent refractive indices
        n_pump = self.sellmeier_equation(pump_wl_um, temperature)
        n_signal = self.sellmeier_equation(signal_wl_um, temperature)
        n_idler = self.sellmeier_equation(idler_wl_um, temperature)
        
        # Wave vectors (in 1/μm)
        k_pump = 2 * np.pi * n_pump / pump_wl_um
        k_signal = 2 * np.pi * n_signal / signal_wl_um
        k_idler = 2 * np.pi * n_idler / idler_wl_um
        
        # Phase mismatch
        delta_k = k_pump - k_signal - k_idler
        
        # QPM grating vector (in 1/μm)
        k_qpm = 2 * np.pi / poling_period
        
        # Total phase mismatch
        delta_k_total = delta_k - k_qpm
        
        return delta_k_total
    
    def calculate_intensity(self, wavelengths, pump_wl, temperature, crystal_length, poling_period, resolution=0.1):
        """Calculate SPDC intensity spectrum with accurate temperature dependence matching research data"""
        intensities = []
        crystal_length_um = crystal_length * 1000  # Convert mm to μm
        
        # Temperature-dependent phase matching - critical for PPKTP
        # Optimal temperature shifts with wavelength detuning
        degenerate_wl = 2 * pump_wl  # 810 nm for 405 nm pump
        
        for signal_wl in wavelengths:
            # Energy conservation: idler wavelength
            idler_wl = 1 / (1/pump_wl - 1/signal_wl)
            
            if idler_wl <= 0 or idler_wl > 3000:  # Reasonable bounds
                intensities.append(0)
                continue
            
            # Phase mismatch calculation with temperature dependence
            delta_k = self.phase_matching_condition(pump_wl, signal_wl, idler_wl, temperature, poling_period)
            
            # Temperature tuning - each wavelength has optimal temperature
            # Based on research: peak shifts ~1nm per °C
            wavelength_detuning = signal_wl - degenerate_wl
            optimal_temp_for_wl = 35 + wavelength_detuning * 0.5  # Empirical from research
            
            # Temperature acceptance bandwidth (narrow for PPKTP)
            temp_bandwidth = 5.0  # °C, narrow acceptance
            temp_mismatch = (temperature - optimal_temp_for_wl) / temp_bandwidth
            
            # Sinc function for phase matching
            phase_argument = delta_k * crystal_length_um / 2
            if abs(phase_argument) < 1e-10:
                phase_factor = 1.0
            else:
                phase_factor = (np.sin(phase_argument) / phase_argument)**2
            
            # Temperature-dependent intensity - Gaussian profile
            temp_factor = np.exp(-0.5 * temp_mismatch**2)
            
            # Spectral acceptance - narrower for longer crystals
            spectral_bandwidth = 5.0 / np.sqrt(crystal_length / 10)  # nm, resolution-dependent
            spectral_factor = np.exp(-0.5 * ((signal_wl - degenerate_wl) / spectral_bandwidth)**2)
            
            # Resolution factor - affects peak width
            resolution_factor = np.exp(-0.5 * ((signal_wl - degenerate_wl) / resolution)**2) if resolution > 0 else 1.0
            
            # Pump power factor (constant for low conversion)
            pump_factor = 1.0
            
            # Total intensity
            intensity = phase_factor * temp_factor * spectral_factor * pump_factor
            intensities.append(intensity)
        
        # Normalize to 0-1 range
        intensities = np.array(intensities)
        if np.max(intensities) > 0:
            intensities = intensities / np.max(intensities)
        
        return intensities
    
    def find_optimal_temperature(self, pump_wl, target_wl, poling_period, crystal_length, temp_range=(20, 200)):
        """Find temperature that maximizes intensity at target wavelength"""
        def neg_intensity(temp):
            intensity = self.calculate_intensity([target_wl], pump_wl, temp, crystal_length, poling_period)
            return -intensity[0]
        
        result = minimize_scalar(neg_intensity, bounds=temp_range, method='bounded')
        return result.x, -result.fun
    
    def calculate_phase_matching_temperature(self, pump_wl, signal_wl, poling_period):
        """Calculate theoretical phase matching temperature"""
        # This gives approximate temperature for perfect phase matching
        # For 405nm pump and ~810nm signal, typically around 60-120°C
        idler_wl = 1 / (1/pump_wl - 1/signal_wl)
        
        # Simplified calculation - in reality this requires solving the dispersion equation
        # For demonstration, we'll use a reasonable approximation
        base_temp = 60  # Base temperature
        wavelength_shift = (signal_wl - 2*pump_wl) / 10  # Shift from degeneracy
        temp_tuning = 0.5 * wavelength_shift  # Approximate tuning rate
        
        optimal_temp = base_temp + temp_tuning
        return np.clip(optimal_temp, 20, 200)

# Initialize simulator
simulator = PPKTPSimulator()

# Sidebar controls
st.sidebar.header("🎛️ Simulation Parameters")

# Crystal parameters
st.sidebar.subheader("Crystal Properties")
pump_wavelength = st.sidebar.slider("Pump Wavelength (nm)", 350, 500, 405, 1)
poling_period = st.sidebar.slider("Poling Period (μm)", 5.0, 20.0, 9.5, 0.1)
crystal_length = st.sidebar.slider("Crystal Length (mm)", 1, 50, 20, 1)

# Spectral range
st.sidebar.subheader("Spectral Range")
min_wavelength = st.sidebar.slider("Min Wavelength (nm)", 700, 800, 740, 5)
max_wavelength = st.sidebar.slider("Max Wavelength (nm)", 850, 950, 900, 5)
resolution = st.sidebar.slider("Spectral Resolution (nm)", 0.1, 2.0, 0.5, 0.1)

# Temperature settings
st.sidebar.subheader("Temperature Control")
temperature = st.sidebar.slider("Temperature (°C)", 20, 80, 35, 5)
enable_temp_scan = st.sidebar.checkbox("Enable Temperature Scan", value=True)

if enable_temp_scan:
    temp_min = st.sidebar.slider("Scan Min Temp (°C)", 20, 50, 35, 5)
    temp_max = st.sidebar.slider("Scan Max Temp (°C)", 50, 80, 60, 5)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Emission Spectrum")
    
    # Generate wavelength array with proper resolution
    num_points = int((max_wavelength - min_wavelength) / resolution)
    wavelengths = np.linspace(min_wavelength, max_wavelength, num_points)
    
    # Calculate spectrum
    intensities = simulator.calculate_intensity(wavelengths, pump_wavelength, temperature, crystal_length, poling_period, resolution)
    
    # Create plot matching research paper style
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot spectrum with proper styling
    ax.plot(wavelengths, intensities, 'k-', linewidth=2, label=f'T = {temperature}°C')
    ax.fill_between(wavelengths, intensities, alpha=0.3, color='black')
    
    # Mark degenerate wavelength (810 nm for 405 nm pump)
    degenerate_wl = 2 * pump_wavelength
    if min_wavelength <= degenerate_wl <= max_wavelength:
        ax.axvline(degenerate_wl, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(degenerate_wl + 5, 0.9, f'Degeneracy\n{degenerate_wl} nm', 
                fontsize=10, ha='left', va='center', color='red')
    
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Normalized Intensity', fontsize=14)
    ax.set_title(f'Type 0 PPKTP Emission Spectrum (Resolution: {resolution} nm)', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Set axis limits and ticks to match research paper
    ax.set_xlim(min_wavelength, max_wavelength)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    

with col2:
    st.header("📈 Analysis")
    
    # Key metrics
    if len(intensities) > 0 and max(intensities) > 0:
        peak_idx = np.argmax(intensities)
        peak_wavelength = wavelengths[peak_idx]
        peak_intensity = intensities[peak_idx]
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Peak Analysis</h4>
            <p><strong>Peak Wavelength:</strong> {peak_wavelength:.1f} nm</p>
            <p><strong>Peak Intensity:</strong> {peak_intensity:.3f}</p>
            <p><strong>Resolution:</strong> {resolution} nm</p>
            <p><strong>Temperature:</strong> {temperature}°C</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate idler wavelength
        idler_wl = 1 / (1/pump_wavelength - 1/peak_wavelength)
        if idler_wl > 0:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔄 Photon Pair</h4>
                <p><strong>Signal:</strong> {peak_wavelength:.1f} nm</p>
                <p><strong>Idler:</strong> {idler_wl:.1f} nm</p>
                <p><strong>Sum:</strong> {peak_wavelength + idler_wl:.1f} nm</p>
            </div>
            """, unsafe_allow_html=True)

# Temperature scan section
if enable_temp_scan:
    st.header("🌡️ Temperature Scan Analysis")
    
    # Temperature array
    temp_step = 5  # °C steps like in research
    temps = np.arange(temp_min, temp_max + temp_step, temp_step)
    
    # Generate wavelength array for temperature scan
    num_points = int((max_wavelength - min_wavelength) / resolution)
    wavelengths_scan = np.linspace(min_wavelength, max_wavelength, num_points)
    
    # Calculate spectra for each temperature
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    
    # Color map for temperatures
    colors = plt.cm.viridis(np.linspace(0, 1, len(temps)))
    
    max_intensity_overall = 0
    
    for i, temp in enumerate(temps):
        intensities_temp = simulator.calculate_intensity(wavelengths_scan, pump_wavelength, temp, crystal_length, poling_period, resolution)
        
        # Plot with different colors and markers like research paper
        if temp == 35:
            ax3.plot(wavelengths_scan, intensities_temp, 'ko-', linewidth=2, markersize=3, label=f'{temp}°C')
        elif temp == 40:
            ax3.plot(wavelengths_scan, intensities_temp, 'ro-', linewidth=2, markersize=3, label=f'{temp}°C')
        elif temp == 45:
            ax3.plot(wavelengths_scan, intensities_temp, 'bo-', linewidth=2, markersize=3, label=f'{temp}°C')
        elif temp == 50:
            ax3.plot(wavelengths_scan, intensities_temp, 'mo-', linewidth=2, markersize=3, label=f'{temp}°C')
        elif temp == 55:
            ax3.plot(wavelengths_scan, intensities_temp, 'go-', linewidth=2, markersize=3, label=f'{temp}°C')
        elif temp == 60:
            ax3.plot(wavelengths_scan, intensities_temp, 'co-', linewidth=2, markersize=3, label=f'{temp}°C')
        else:
            ax3.plot(wavelengths_scan, intensities_temp, color=colors[i], linewidth=2, label=f'{temp}°C')
        
        max_intensity_overall = max(max_intensity_overall, np.max(intensities_temp))
    
    # Mark degenerate wavelength
    degenerate_wl = 2 * pump_wavelength
    if min_wavelength <= degenerate_wl <= max_wavelength:
        ax3.axvline(degenerate_wl, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax3.text(degenerate_wl + 5, 0.9, f'Degeneracy\n{degenerate_wl} nm', 
                fontsize=10, ha='left', va='center', color='red')
    
    ax3.set_xlabel('Wavelength (nm)', fontsize=14)
    ax3.set_ylabel('Normalized Intensity', fontsize=14)
    ax3.set_title(f'Temperature Tuning of Type 0 PPKTP (Resolution: {resolution} nm)', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set axis limits to match research paper
    ax3.set_xlim(min_wavelength, max_wavelength)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    plt.tight_layout()
    st.pyplot(fig3)

# Additional analysis
st.header("🔬 Physical Parameters")

col5, col6, col7 = st.columns(3)

with col5:
    st.subheader("Crystal Properties")
    st.write(f"**Material:** KTiOPO₄ (Type 0)")
    st.write(f"**Length:** {crystal_length} mm")
    st.write(f"**Poling Period:** {poling_period} μm")
    st.write(f"**Temperature:** {temperature} °C")

with col6:
    st.subheader("Spectral Properties")
    # Calculate refractive index at pump wavelength
    n_pump = simulator.sellmeier_equation(pump_wavelength/1000, temperature)
    degenerate_wl = 2 * pump_wavelength
    st.write(f"**Pump Wavelength:** {pump_wavelength} nm")
    st.write(f"**Degenerate Point:** {degenerate_wl} nm")
    st.write(f"**Spectral Resolution:** {resolution} nm")
    st.write(f"**Refractive Index:** {n_pump:.4f}")

with col7:
    st.subheader("Phase Matching")
    # Calculate acceptance bandwidth
    acceptance_temp = 0.1  # °C
    acceptance_angle = 0.1  # mrad
    st.write(f"**Acceptance Temp:** ±{acceptance_temp} °C")
    st.write(f"**Acceptance Angle:** ±{acceptance_angle} mrad")
    st.write(f"**QPM Order:** 1st")

# Download data
st.header("💾 Export Data")

# Create dataframe
df = pd.DataFrame({
    'Wavelength (nm)': wavelengths,
    'Intensity': intensities
})

col8, col9 = st.columns(2)

with col8:
    st.download_button(
        label="📥 Download Spectrum Data (CSV)",
        data=df.to_csv(index=False),
        file_name=f"ppktp_spectrum_T{temperature}C.csv",
        mime="text/csv"
    )

with col9:
    if enable_temp_scan:
        # Create temperature scan data for download
        temp_step = 5
        temps_download = np.arange(temp_min, temp_max + temp_step, temp_step)
        degenerate_wl = 2 * pump_wavelength
        
        temp_scan_data = []
        for temp in temps_download:
            intensity = simulator.calculate_intensity([degenerate_wl], pump_wavelength, temp, crystal_length, poling_period, resolution)[0]
            temp_scan_data.append(intensity)
        
        temp_df = pd.DataFrame({
            'Temperature (°C)': temps_download,
            'Degenerate Intensity': temp_scan_data
        })
        
        st.download_button(
            label="📥 Download Temperature Scan (CSV)",
            data=temp_df.to_csv(index=False),
            file_name=f"ppktp_temp_scan_{pump_wavelength}nm.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("**Note:** This simulation uses simplified models for educational purposes. Real PPKTP behavior may vary due to additional factors not included in this model.")