import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats

# Load data from a CSV file
data = pd.read_csv('BlueFCF.csv')

# Extract wavelength and intensity data
wavelength = np.array(data['nm lambda'])
intensities = data.iloc[:, 1:].values

# Filter data for specific wavelengths
mask = (wavelength > 450) & (wavelength < 750)
wavelength_filtered = wavelength[mask]
intensities_filtered = intensities[mask, :]
sample_count = intensities_filtered.shape[1]

# Data overview
print(f"Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} nm")
print(f"Number of samples: {sample_count}")
print(f"Data shape: {intensities.shape}")

# Define concentrations and labels
c1 = (3/32) * 0.000225137  # 1st solution - most concentrated
c1 = 1
concentrations = np.array([3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10, 1])
labels = ['c₁', '9c₁/10', '8c₁/10', '7c₁/10', '6c₁/10', '5c₁/10', '4c₁/10', '3c₁/10']


# Define figure and colors for plotting
plt.figure(figsize=(11, 6))
colors = plt.cm.plasma(np.linspace(0.3, 0.9, 8))

# plot the absorption spectra
for i in range(8):
    plt.plot(wavelength_filtered, intensities_filtered[:, i], label=labels[i], color=colors[i], linewidth=2, alpha=0.8)
    
# Label the plots
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Absorbance (natural units)', fontsize=12)
plt.title('Brilliant Blue FCF Absorption Spectra at Different Concentrations', fontsize=13)
plt.legend(title='Concentration', ncol=2)
plt.grid(alpha=0.3)
plt.xlim(wavelength_filtered.min(), wavelength_filtered.max())
plt.tight_layout()
plt.show()


# Find peaks in all spectras
absorption_peaks = []
wavelength_peaks = []
for sample in range(sample_count):
    print(sample)
    peak_index = intensities_filtered[:, sample].argmax()
    peak_absorption = intensities_filtered[:, sample][peak_index]
    peak_wavelength = wavelength_filtered[peak_index]

    absorption_peaks.append(peak_absorption)
    wavelength_peaks.append(peak_wavelength)

    """
    print(peak_index)
    print(peak_absorption)
    print(peak_wavelength)
    print('---')
    """


# Plot peak absorption vs concentration
plt.figure(figsize=(8, 5))
plt.scatter(concentrations, absorption_peaks, color='red', s=40, label='Data points')

# Perform linear regression
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(concentrations, absorption_peaks)

# Plot line of best fit
fit_line = slope * concentrations + intercept
plt.plot(concentrations, fit_line, linestyle='--', linewidth=2, label=f'Linear fit: y = {slope:.2f}x + {intercept:.2f}\n')

plt.xlabel('Concentration (moles / litre)', fontsize=12)
plt.ylabel('Peak Absorbance (natural units)', fontsize=12)
plt.title('Peak Absorbance vs Concentration of Brilliant Blue FCF', fontsize=13)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
