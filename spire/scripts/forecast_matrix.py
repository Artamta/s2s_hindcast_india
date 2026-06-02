import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe for cluster compute nodes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# 1. Setup Output Directory
OUTPUT_DIR = "paper_figures_2026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Connect to Arraylake & Target the ANOMALIES Group
print("Connecting to Arraylake [anomalies group]...")
from arraylake import Client
client = Client()
repo = client.get_repo("artamta/s2s-research")
session = repo.readonly_session("main")

# Open the anomalies group directly
ds = xr.open_zarr(session.store, group="anomalies")

# 3. Spatial Slicing for India Domain
lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

# Handle inverted latitude coordinates safely
lat_slice = slice(38.0, 7.0) if ds[lat_name][0] > ds[lat_name][-1] else slice(7.0, 38.0)
lon_slice = slice(65.0, 98.0)
ds_india = ds.sel({lat_name: lat_slice, lon_name: lon_slice})

# Extract 2m Maximum Temperature Anomaly (Units: Kelvin deviation = Celsius deviation value)
t_anomaly = ds_india['air_temperature_max']

# 4. Select an Initialization Date
# Lock onto Index 40 (February 10, 2026) to track the anomalies
REF_INDEX = 40 
ref_date_actual = str(ds_india['reference_time'].isel(reference_time=REF_INDEX).values)[:10]
print(f"Processing true weekly-averaged anomalies for initialization: {ref_date_actual}...")

# Compute 7-day windows using .isel() array index boundaries
week_blocks = [
    {'name': 'Week 1 Horizon (Days 1-7)',   'slice': slice(0, 7)},
    {'name': 'Week 2 Horizon (Days 8-14)',  'slice': slice(7, 14)},
    {'name': 'Week 3 Horizon (Days 15-21)', 'slice': slice(14, 21)},
    {'name': 'Week 4 Horizon (Days 22-28)', 'slice': slice(21, 28)}
]

# Extract time slice and calculate weekly means
weekly_data = []
for block in week_blocks:
    mean_slice = t_anomaly.isel(reference_time=REF_INDEX).isel(step=block['slice']).mean(dim='step').compute()
    weekly_data.append(mean_slice)

# 5. Build the Canvas Matrix
fig, axes = plt.subplots(2, 2, figsize=(14, 13), 
                         subplot_kw={'projection': ccrs.PlateCarree()}, 
                         gridspec_kw={'wspace': 0.12, 'hspace': 0.18}, 
                         dpi=300)
axes = axes.flatten()

# Establish balanced symmetric boundaries for the anomaly scale (e.g., -5°C to +5°C)
vmax = max(abs(float(d.max())) for d in weekly_data)
vmax = min(max(vmax, 3.0), 6.0) # Keep scale realistic and standardized
vmin = -vmax

# Loop through panels
for i, block in enumerate(week_blocks):
    ax = axes[i]
    data = weekly_data[i]
    
    # Base geography map layers
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#eef1f6', zorder=1)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#fdfefe', zorder=0)
    
    # CRITICAL FIX: add_labels=False turns off Xarray's auto-generated metadata strings
    im = data.plot(ax=ax, transform=ccrs.PlateCarree(), 
                   cmap='RdYlBu_r', vmin=vmin, vmax=vmax, 
                   add_colorbar=False, add_labels=False, zorder=2)
    
    # Force-clear any rogue default titles that Xarray tries to inject anyway
    ax.set_title("")
    
    # Add crisp, standardized borders
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.1, edgecolor='#2c3e50', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.6, edgecolor='#7f8c8d', linestyle=':', zorder=3)
    ax.set_extent([66.0, 97.0, 8.0, 37.0], crs=ccrs.PlateCarree())
    
    # Gridline aesthetics
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color='gainsboro', alpha=0.7, linestyle='--')
    gl.top_labels, gl.right_labels = False, False
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 9, 'color': '#2c3e50'}
    gl.ylabel_style = {'size': 9, 'color': '#2c3e50'}
    
    # Apply clean custom titles with an increased, protective pad parameter
    ax.set_title(block['name'], fontsize=12, fontweight='bold', pad=14, loc='left')

# 6. Consolidated Shared Horizontal Colorbar
cbar_ax = fig.add_axes([0.22, 0.06, 0.56, 0.022])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label("Maximum Temperature Anomaly Value (°C deviation from ERA5 Climatology)", fontsize=11, fontweight='bold', labelpad=6)
cbar.ax.tick_params(labelsize=10)

# Main figure title - set with y parameter to clear the top row subplots completely
fig.suptitle(f"Spire S2S Subseasonal Temperature Evolution Matrix\nBase Initialization Date: {ref_date_actual}", 
             fontsize=15, fontweight='bold', y=0.97)

# Save high-resolution figure with strict bounding box containment
output_name = f"{OUTPUT_DIR}/figure3_temperature_anomalies_evolution.png"
plt.savefig(output_name, bbox_inches='tight', dpi=300)
plt.close()

print(f"\nSuccess! Subseasonal evolution anomaly matrix saved to: {output_name}")