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

# 2. Connect to Arraylake and fetch anomalies group
print("Connecting to Arraylake [anomalies group]...")
from arraylake import Client
client = Client()
repo = client.get_repo("artamta/s2s-research")
session = repo.readonly_session("main")

ds_anom = xr.open_zarr(session.store, group="anomalies")

# 3. Select a specific reference day and lead steps to track
# We will use initialization index 40 (February 10, 2026)
REF_IDX = 40
ref_date = str(ds_anom['reference_time'].isel(reference_time=REF_IDX).values)[:10]

# Choose 4 specific days into the future to track forecast evolution
# Array indices: 0 = Day 1, 4 = Day 5, 9 = Day 10, 14 = Day 15
lead_indices = [0, 4, 9, 14]
lead_days = [1, 5, 10, 15]

print(f"Extracting global snapshot sequence for Init: {ref_date}...")

# 4. Set up a 2x2 Global Matrix Grid (Robinson Projection)
fig, axes = plt.subplots(2, 2, figsize=(18, 11), 
                         subplot_kw={'projection': ccrs.Robinson(central_longitude=0)},
                         gridspec_kw={'wspace': 0.08, 'hspace': 0.12}, dpi=300)
axes = axes.flatten()

# Dynamic symmetric color boundaries for global anomalies
vmax = 7.0
vmin = -vmax

# Loop through our chosen lead days
for i, step_idx in enumerate(lead_indices):
    ax = axes[i]
    day = lead_days[i]
    
    # Extract global snapshot for this specific day
    data = ds_anom['air_temperature_max'].isel(reference_time=REF_IDX, step=step_idx).compute()
    
    # Base geography layers
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='#f4f4f4', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f2f1', zorder=0)
    
    # Plot global anomalies (Divergent colormap: Blue=Cold, White=Normal, Red=Hot)
    im = data.plot(ax=ax, transform=ccrs.PlateCarree(),
                   cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                   add_colorbar=False, add_labels=False, zorder=1)
    
    # Structural details
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#444444', zorder=2)
    ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.3, color='gray', alpha=0.4, linestyle='--')
    
    # Subplot Title
    ax.set_title(f"Forecast Day +{day} (Valid Date: Feb {10+day}, 2026)", fontsize=12, fontweight='bold', pad=8)

# 5. Shared Global Horizontal Colorbar
cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.022])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label("Predicted Max Temperature Anomaly (°C deviation from 30-year historical mean)", fontsize=11, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Main Title
fig.suptitle(f"Global Forecast Evolution Sequence\nBase Initialization Date: {ref_date}", 
             fontsize=16, fontweight='bold', y=0.96)

# Save global sequence image
output_path = f"{OUTPUT_DIR}/global_forecast_tracking_sequence.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Success! Global tracking sequence saved to: {output_path}")