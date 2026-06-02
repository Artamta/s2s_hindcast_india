import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe for cluster environments
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 1. Setup Output Directory
OUTPUT_DIR = "paper_figures_2026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Connect to Arraylake and fetch groups
print("Connecting to Arraylake...")
from arraylake import Client
client = Client()
repo = client.get_repo("artamta/s2s-research")
session = repo.readonly_session("main")

# Open both groups to compare signal vs. uncertainty
ds_anom = xr.open_zarr(session.store, group="anomalies")
ds_std  = xr.open_zarr(session.store, group="mean_stddev")

# 3. Select a specific slice in time (Global context)
# Let's look at Initialization Index 40, Lead Time Step Index 14 (Week 2/Day 15 forecast)
REF_IDX = 40
STEP_IDX = 14

ref_date = str(ds_anom['reference_time'].isel(reference_time=REF_IDX).values)[:10]
lead_days = int(ds_anom['step'].isel(step=STEP_IDX).values / np.timedelta64(1, 'D'))

print(f"Extracting global maps for Init: {ref_date} | Lead Time: {lead_days} Days...")

# Extract global variables
# Left: Max Temperature Anomaly
anomaly_data = ds_anom['air_temperature_max'].isel(reference_time=REF_IDX, step=STEP_IDX).compute()
# Right: Ensemble Standard Deviation of Air Temperature
spread_data = ds_std['air_temperature_stddev'].isel(reference_time=REF_IDX, step=STEP_IDX).compute()

# 4. Set up a global 1x2 Robinson Projection Canvas (Standard for world maps)
fig, axes = plt.subplots(1, 2, figsize=(20, 8), 
                         subplot_kw={'projection': ccrs.Robinson(central_longitude=0)},
                         gridspec_kw={'wspace': 0.15}, dpi=300)

# --- PANEL 1: THE ANOMALY SIGNAL ---
ax1 = axes[0]
ax1.set_global()
ax1.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=0)
ax1.add_feature(cfeature.OCEAN, facecolor='#e0f2f1', zorder=0)

# Balanced color limits for anomaly mapping
vmax_anom = 8.0 
im1 = anomaly_data.plot(ax=ax1, transform=ccrs.PlateCarree(),
                        cmap='RdYlBu_r', vmin=-vmax_anom, vmax=vmax_anom,
                        add_colorbar=False, add_labels=False, zorder=1)

ax1.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#333333', zorder=2)
ax1.gridlines(crs=ccrs.PlateCarree(), linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
ax1.set_title(f"(a) Predicted Max Temperature Anomaly\n(Deviation from ERA5 Baseline)", fontsize=13, fontweight='bold', pad=10)

# Colorbar 1
cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7, extend='both')
cbar1.set_label("Temperature Anomaly (°C or K deviation)", fontsize=11)

# --- PANEL 2: THE ENSEMBLE SPREAD (UNCERTAINTY) ---
ax2 = axes[1]
ax2.set_global()
ax2.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=0)
ax2.add_feature(cfeature.OCEAN, facecolor='#eef1f6', zorder=0)

# Sequential color limits for standard deviation (always positive values)
im2 = spread_data.plot(ax=ax2, transform=ccrs.PlateCarree(),
                       cmap='magma_r', vmin=0, vmax=5.0,
                       add_colorbar=False, add_labels=False, zorder=1)

ax2.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#333333', zorder=2)
ax2.gridlines(crs=ccrs.PlateCarree(), linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
ax2.set_title(f"(b) Ensemble Standard Deviation\n(Model Uncertainty / Disagreement)", fontsize=13, fontweight='bold', pad=10)

# Colorbar 2
cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.7, extend='max')
cbar2.set_label("Standard Deviation Among Members (K)", fontsize=11)

# Main Title
fig.suptitle(f"Spire Global S2S Diagnostics\nInitialization Base Date: {ref_date} | Forecast Lead: +{lead_days} Days", 
             fontsize=16, fontweight='bold', y=0.98)

# Save image
output_path = f"{OUTPUT_DIR}/global_s2s_diagnostic_map.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Success! Global diagnostic map saved to: {output_path}")