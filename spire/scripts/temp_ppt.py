import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Safe for cluster compute nodes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

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

# 3. Spatial Slicing for India Domain
lat_name = 'latitude' if 'latitude' in ds_anom.coords else 'lat'
lon_name = 'longitude' if 'longitude' in ds_anom.coords else 'lon'

lat_slice = slice(38.0, 7.0) if ds_anom[lat_name][0] > ds_anom[lat_name][-1] else slice(7.0, 38.0)
lon_slice = slice(65.0, 98.0)
ds_india = ds_anom.sel({lat_name: lat_slice, lon_name: lon_slice})

# Extract both variables for a key subseasonal lead time (e.g., Week 2 mean, days 8-14)
print("Extracting Week 2 temperature and precipitation fields...")
t_anom = ds_india['air_temperature_max'].isel(step=slice(7, 14)).mean(dim='step')
p_anom = ds_india['precipitation_amount'].isel(step=slice(7, 14)).mean(dim='step')

# 4. Compute Gridpoint-by-Gridpoint Pearson Correlation Coefficient (r)
print("Computing spatial co-variability correlation matrix across all initializations...")

# Center the data over the reference_time dimension to compute covariance safely
t_prime = t_anom - t_anom.mean(dim='reference_time')
p_prime = p_anom - p_anom.mean(dim='reference_time')

covariance = (t_prime * p_prime).mean(dim='reference_time')
t_std = t_anom.std(dim='reference_time')
p_std = p_anom.std(dim='reference_time')

# Pearson correlation formula: r = cov(X,Y) / (std(X) * std(Y))
# Avoid division by zero bugs with a tiny epsilon
r_matrix = (covariance / (t_std * p_std + 1e-6)).compute()

# 5. Build a Highly Polished Map
print("Generating spatial correlation plot...")
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)

# Base geography layers
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#eef1f6', zorder=1)
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#fdfefe', zorder=0)

# Plot correlation matrix (Divergent colormap: Brown/BrBG means anti-correlated, green means positively coupled)
im = r_matrix.plot(ax=ax, transform=ccrs.PlateCarree(),
                   cmap='BrBG', vmin=-0.8, vmax=0.8,
                   add_colorbar=False, zorder=2)

# Clear Xarray default string metadata completely
ax.set_title("")

# Add sharp borders
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='#2c3e50', zorder=3)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.7, edgecolor='#7f8c8d', linestyle=':', zorder=3)
ax.set_extent([66.0, 97.0, 8.0, 37.0], crs=ccrs.PlateCarree())

# Gridline formatting
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color='gainsboro', alpha=0.7, linestyle='--')
gl.top_labels, gl.right_labels = False, False
gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
gl.xlabel_style = {'size': 9, 'color': '#2c3e50'}
gl.ylabel_style = {'size': 9, 'color': '#2c3e50'}

# Custom title
ax.set_title("Week 2 Co-variability: $T_{max}$ vs. Precipitation Anomaly Correlation", 
             fontsize=13, fontweight='bold', pad=12, loc='left')

# Shared Colorbar
cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.06, shrink=0.75, extend='both')
cbar.set_label("Pearson Correlation Coefficient ($r$)", fontsize=10, fontweight='bold', labelpad=6)
cbar.ax.tick_params(labelsize=9)

# Save high-resolution chart
output_name = f"{OUTPUT_DIR}/figure5_temperature_precipitation_correlation.png"
plt.savefig(output_name, bbox_inches='tight', dpi=300)
plt.close()

print(f"\nSuccess! Spatial feedback correlation map saved to: {output_name}")