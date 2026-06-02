import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 1. Setup Output Directory for India Sequence
OUTPUT_DIR = "india_daily_lifecycle"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Connect to Arraylake
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

# Select an Initialization Date to view (e.g., Index 40)
REF_IDX = 40
ref_date = str(ds_anom['reference_time'].isel(reference_time=REF_IDX).values)[:10]

print(f"Generating India regional daily consecutive sequence for initialization: {ref_date}")

# 4. Loop day-by-day through the forecast lifespan
for step_idx in range(45):
    lead_day = int(ds_anom['step'].isel(step=step_idx).values / np.timedelta64(1, 'D'))
    
    print(f"  -> Processing India Frame: Day {lead_day}")
    
    # Slice the spatial domain for this single forecast step
    india_daily = ds_anom['air_temperature_max'].sel({lat_name: lat_slice, lon_name: lon_slice}).isel(reference_time=REF_IDX, step=step_idx).compute()
    
    # Plotting regional canvas
    fig = plt.figure(figsize=(7, 7), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='#fbfcfc', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#eef1f6', zorder=0)
    
    im = india_daily.plot(ax=ax, transform=ccrs.PlateCarree(),
                         cmap='RdYlBu_r', vmin=-4, vmax=4,
                         add_colorbar=False, add_labels=False, zorder=1)
    
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#2c3e50', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#7f8c8d', linestyle=':', zorder=2)
    ax.set_extent([66.0, 97.0, 8.0, 37.0], crs=ccrs.PlateCarree())
    
    # Frame Title tracking daily lifecycle
    ax.set_title(f"India Regional Forecast  |  Init Base: {ref_date}\nTimeline Evolution: Day +{lead_day}", 
              fontsize=11, fontweight='bold', pad=10)
    
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.07, shrink=0.7, extend='both')
    cbar.set_label("Temperature Anomaly (°C)", fontsize=9)
    
    # Save individual sequential frame
    plt.savefig(f"{OUTPUT_DIR}/india_frame_day_{lead_day:02d}.png", bbox_inches='tight')
    plt.close()

print(f"\nSuccess! All regional frames saved inside the folder: /{OUTPUT_DIR}")