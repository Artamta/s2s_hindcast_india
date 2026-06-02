import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Safe for cluster compute nodes
import matplotlib.pyplot as plt

# 1. Setup Output Directory
OUTPUT_DIR = "paper_figures_2026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Connect to Arraylake [anomalies group]
print("Connecting to Arraylake [anomalies group]...")
from arraylake import Client
client = Client()
repo = client.get_repo("artamta/s2s-research")
session = repo.readonly_session("main")
ds_anom = xr.open_zarr(session.store, group="anomalies")

# 3. Spatial Slicing for India Domain
lat_name = 'latitude' if 'latitude' in ds_anom.coords else 'lat'
lon_name = 'longitude' if 'longitude' in ds_anom.coords else 'lon'
lat_slice = slice(38.0, 7.0) if ds_anom[lat_name][0] > ds_anom[lat_name][-1] else slice(7.0, 38.0)
lon_slice = slice(65.0, 98.0)
ds_india = ds_anom.sel({lat_name: lat_slice, lon_name: lon_slice})

# Extract target variable
t_anom = ds_india['air_temperature_max']

print("Calculating data summaries for simple profile...")
# Pick 3 spread-out initialization dates to compare different forecast paths
# Index 0 (Jan 1), Index 40 (Feb 10), Index 75 (Mid-March)
init_indices = [0, 40, 75]
init_dates = [str(ds_india['reference_time'].isel(reference_time=idx).values)[:10] for idx in init_indices]

# Calculate the spatial mean over India for these 3 dates across all 46 lead days
spatial_means = []
for idx in init_indices:
    mean_series = t_anom.isel(reference_time=idx).mean(dim=[lat_name, lon_name]).compute()
    spatial_means.append(mean_series)

# Convert lead time step coordinates to integer days
lead_days = ds_anom['step'].values / np.timedelta64(1, 'D')

# Extract a large block of individual grid point values to show data distribution
hist_data = t_anom.isel(reference_time=40).values.flatten()
hist_data = hist_data[~np.isnan(hist_data)]  # Clean out any NaN padding

# 4. Create a 1x2 Profiling Canvas
print("Plotting simple data analysis profile...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# --- PANEL 1: THE HISTOGRAM (What is the data spread?) ---
ax1.hist(hist_data, bins=50, color='#3498db', edgecolor='#2980b9', alpha=0.75, rwidth=0.85)
ax1.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Climate Normal (0°C Anomaly)')
ax1.set_title("(a) Distribution of Temperature Anomalies over India", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("Anomaly Value (°C deviation from normal)", fontsize=10, fontweight='bold')
ax1.set_ylabel("Frequency (Total Data Grid Points)", fontsize=10, fontweight='bold')
ax1.grid(color='gainsboro', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.legend(frameon=True, facecolor='white', edgecolor='none')

# --- PANEL 2: THE FORECAST TRAJECTORIES (How do timelines work?) ---
colors = ['#27ae60', '#d35400', '#8e44ad']
for i, mean_series in enumerate(spatial_means):
    ax2.plot(lead_days, mean_series, color=colors[i], linewidth=2, marker='o', markersize=3,
             label=f"Start Date: {init_dates[i]}")

ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax2.set_title("(b) 46-Day Forecast Trajectories (India Average)", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel("Forecast Lead Time (Days into the future)", fontsize=10, fontweight='bold')
ax2.set_ylabel("Domain-Average Anomaly (°C)", fontsize=10, fontweight='bold')
ax2.set_xlim(1, 46)
ax2.grid(color='gainsboro', linestyle='--', linewidth=0.5, alpha=0.5)
ax2.legend(frameon=True, facecolor='white', edgecolor='none')

# Clean frame aesthetics
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Main Figure Title
fig.suptitle("Spire S2S Dataset Profile: Simple Data Audit & Behavior Check", 
             fontsize=15, fontweight='bold', y=1.01)

# Save high-resolution chart
output_name = f"{OUTPUT_DIR}/figure6_simple_data_profile.png"
plt.savefig(output_name, bbox_inches='tight', dpi=300)
plt.close()

print(f"Success! Data profile plot saved to: {output_name}")