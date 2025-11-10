import pandas as pd
import matplotlib.pyplot as plt

# Load both files, skipping metadata
def load_spectrum(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Find start of spectral data
    start_idx = next(i for i, line in enumerate(lines) if 'Begin Spectral Data' in line) + 1
    # Read numeric data into DataFrame
    data = pd.read_csv(filename, sep=r'\s+', skiprows=start_idx, names=['Wavelength', 'Intensity'], comment='>')
    return data

# Load spectra
bg = load_spectrum("5s_bg.txt")
ai = load_spectrum("Al__OFX028761__25__12-32-09-804.txt")

# Merge on Wavelength (inner join keeps only matching values)
merged = pd.merge(bg, ai, on='Wavelength', suffixes=('_bg', '_ai'))

# Compute difference in intensities
merged['Difference'] = merged['Intensity_ai'] - merged['Intensity_bg']

# Save result to file
merged[['Wavelength', 'Difference']].to_csv('Al__OFX028761__25__12-32-09-804_difference_output_5sec.txt', sep='\t', index=False)
print(merged[['Wavelength', 'Difference']].head())

# Bar plot
plt.figure(figsize=(10,5))
plt.bar(merged['Wavelength'], merged['Difference'], width=0.5)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity Difference (AI_OF - BG)')
plt.title('Spectral Intensity Difference')
plt.tight_layout()
plt.grid(True)
plt.show()