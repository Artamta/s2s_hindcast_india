import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Setup Directory
OUTPUT_DIR = "basic_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Connect to Arraylake
from arraylake import Client
client = Client()
repo = client.get_repo("artamta/s2s-research")
session = repo.readonly_session("main")
ds_anom = xr.open_zarr(session.store, group="anomalies")

# Choose a March initialization date (Index 75 is mid-March)
# Choose a long-range forecast lead time (Index 39 is Day 40)
REF_IDX = 75
STEP_IDX = 39

ref_date = str(ds_anom['reference_time'].isel(reference_time=REF_IDX).values)[:10]
lead_day = int(ds_anom['step'].isel(step=STEP_IDX).values / np.timedelta64(1, 'D'))

# Extract the global data
global_data = ds_anom['air_temperature_max'].isel(reference_time=REF_IDX, step=STEP_IDX).compute()

# Plot a simple global map
fig = plt.figure(figsize=(12, 6), dpi=300)
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))

ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
ax.add_feature(cfeature.OCEAN, facecolor='#e0f2f1')

# Plot the temperature anomalies
im = global_data.plot(ax=ax, transform=ccrs.PlateCarree(),
                      cmap='RdYlBu_r', vmin=-5, vmax=5,
                      add_colorbar=False, add_labels=False)

ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#333333')

# Simple Title
plt.title(f"Global Forecast Map\nInitialized on: {ref_date} | Showing Day +{lead_day} in the Future", 
          fontsize=12, fontweight='bold', pad=10)

# Simple Colorbar
cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
cbar.set_label("Temperature Anomaly (°C)", fontsize=10)

plt.savefig(f"{OUTPUT_DIR}/1_global_march_simple.png", bbox_inches='tight', dpi=300)
plt.close()
print("Saved: 1_global_march_simple.png")