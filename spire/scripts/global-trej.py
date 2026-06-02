import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 1. Setup Output Directory for the Global Sequence
OUTPUT_DIR = "global_march_daily_sequence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Connect to Arraylake
from arraylake import Client
client = Client()
repo = client.get_repo("artamta/s2s-research")
session = repo.readonly_session("main")
ds_anom = xr.open_zarr(session.store, group="anomalies")

# 3. Select March Reference Date (Index 75)
REF_IDX = 75
ref_date = str(ds_anom['reference_time'].isel(reference_time=REF_IDX).values)[:10]

print(f"Generating global daily consecutive forecast sequence for initialization: {ref_date}")

# 4. Loop through EVERY step index sequentially from Day 1 to Day 45
# (Most S2S models use index 0 as Day 1, index 44 as Day 45)
for step_idx in range(45):
    # Calculate the actual lead day number
    lead_day = int(ds_anom['step'].isel(step=step_idx).values / np.timedelta64(1, 'D'))
    
    print(f"  -> Processing Global Frame: Day {lead_day}")
    
    # Extract the global data array for this specific day
    daily_data = ds_anom['air_temperature_max'].isel(reference_time=REF_IDX, step=step_idx).compute()
    
    # Build a clean single-frame map
    fig = plt.figure(figsize=(12, 6), dpi=150) # Optimized DPI for fast sequential rendering
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f2f1', zorder=0)
    
    # Plot the grid
    im = daily_data.plot(ax=ax, transform=ccrs.PlateCarree(),
                          cmap='RdYlBu_r', vmin=-5, vmax=5,
                          add_colorbar=False, add_labels=False, zorder=1)
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#333333', zorder=2)
    
    # Strict title format highlighting the daily march forward
    ax.set_title(f"Global Subseasonal Forecast  |  Init: {ref_date}\nForecast Horizon: Day +{lead_day} consecutive", 
              fontsize=12, fontweight='bold', pad=10)
    
    # Bottom colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.06, shrink=0.6, extend='both')
    cbar.set_label("Temperature Anomaly (°C)", fontsize=10)
    
    # Save frame with zero padding so they sort perfectly in your file manager (01, 02, 03...)
    plt.savefig(f"{OUTPUT_DIR}/global_frame_day_{lead_day:02d}.png", bbox_inches='tight')
    plt.close()

print(f"\nSuccess! All global daily frames saved inside the folder: /{OUTPUT_DIR}")