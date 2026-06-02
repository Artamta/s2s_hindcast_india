import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Safe for cluster environments
import matplotlib.pyplot as plt

# 1. Setup Output Directory
OUTPUT_DIR = "paper_figures_2026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Connect to Arraylake and fetch anomalies group
print("Connecting to Arraylake [anomalies group]...")
from arraylake import Client
client = Client()
repo = client.get_repo("artamta/s2s-research")
session = repo.readonly_session("main")

ds_anom = xr.open_zarr(session.store, group="anomalies")

# 3. Spatial Slicing for India Domain (Matching your exact matrix map bounds)
lat_name = 'latitude' if 'latitude' in ds_anom.coords else 'lat'
lon_name = 'longitude' if 'longitude' in ds_anom.coords else 'lon'

lat_slice = slice(38.0, 7.0) if ds_anom[lat_name][0] > ds_anom[lat_name][-1] else slice(7.0, 38.0)
lon_slice = slice(65.0, 98.0)

print("Slicing data over Indian subcontinent...")
t_anom_india = ds_anom['air_temperature_max'].sel({lat_name: lat_slice, lon_name: lon_slice})

# 4. Compute Systematic RMSE Skill Decay
print("Calculating spatial RMSE across all initializations and lead times... (This may take a moment)")

# Step A: Square the anomalies to compute magnitude of forecast variations/errors
squared_anom = t_anom_india ** 2

# Step B: Average over the spatial dimensions (latitude and longitude)
spatial_mse = squared_anom.mean(dim=[lat_name, lon_name])

# Step C: Average across all initialization dates (reference_time) to get systematic performance
systematic_mse = spatial_mse.mean(dim='reference_time').compute()

# Step D: Take the square root to return to Kelvin/Celsius scale
rmse_decay = np.sqrt(systematic_mse)

# Convert lead time step coordinate from nanoseconds/timedeltas to actual integer days
lead_days = ds_anom['step'].values / np.timedelta64(1, 'D')

# 5. Plotting a Crisp, Journal-Ready Line Graph
print("Generating skill decay visualization...")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot main decay curve
ax.plot(lead_days, rmse_decay, color='#d35400', linewidth=2.5, linestyle='-', 
        label='Systematic Model Anomaly Amplitude (RMSE)')

# Enhance aesthetics
ax.set_title("S2S Temperature Predictability & Skill Decay Curve over India", 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Forecast Lead Time (Days into the future)", fontsize=11, fontweight='bold', labelpad=8)
ax.set_ylabel("Spatial Root Mean Squared Anomaly Magnitude (°C)", fontsize=11, fontweight='bold', labelpad=8)

# Customize gridlines and structural boundaries
ax.grid(color='gainsboro', linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_xlim(1, 46)
ax.set_xticks(np.append([1], np.arange(5, 47, 5))) # Explicit, clean tick markers

# Add descriptive contextual annotations to help reviewers read the chart
ax.fill_between(lead_days, rmse_decay, color='#fdebd0', alpha=0.3)

# Highlight standard subseasonal split lines (Week 1, Week 2, Week 3/4 boundaries)
ax.axvline(x=7, color='#7f8c8d', linestyle=':', linewidth=1)
ax.axvline(x=14, color='#7f8c8d', linestyle=':', linewidth=1)
ax.text(3.5, ax.get_ylim()[1]*0.05, 'Week 1', fontsize=9, style='italic', color='#555555', ha='center')
ax.text(10.5, ax.get_ylim()[1]*0.05, 'Week 2', fontsize=9, style='italic', color='#555555', ha='center')
ax.text(20.0, ax.get_ylim()[1]*0.05, 'Weeks 3-4 Outlook Horizon', fontsize=9, style='italic', color='#555555', ha='center')

# Polish frame lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#2c3e50')
ax.spines['bottom'].set_color('#2c3e50')

# Save high-resolution chart
output_path = f"{OUTPUT_DIR}/figure4_skill_decay_curve.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\nSuccess! Skill decay curve plot saved to: {output_path}")