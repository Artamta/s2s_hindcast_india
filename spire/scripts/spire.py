import os
import urllib.request
import xarray as xr
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from arraylake import Client 

# =====================================================================
# 1. Setup Folders and Download Official India Shapefile
# =====================================================================
OUTPUT_DIR = "weekly_forecasts_india"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHAPEFILE_DIR = "india_maps"
os.makedirs(SHAPEFILE_DIR, exist_ok=True)
shp_path = os.path.join(SHAPEFILE_DIR, "india_administrative_outline.shp")

# If you don't have a local compliant shapefile, we can load a public validated repo map.
# (If you already have an official .shp from Survey of India or bhuvan, put it in 'india_maps' folder!)
SHP_URL = "https://raw.githubusercontent.com/AnujTiwari/India-State-and-Country-Shapefile/master/India_Country_Boundary.shp"
SHX_URL = "https://raw.githubusercontent.com/AnujTiwari/India-State-and-Country-Shapefile/master/India_Country_Boundary.shx"
DBF_URL = "https://raw.githubusercontent.com/AnujTiwari/India-State-and-Country-Shapefile/master/India_Country_Boundary.dbf"

if not os.path.exists(shp_path):
    print("Downloading compliant India boundary shapefile (including POK)...")
    try:
        urllib.request.urlretrieve(SHP_URL, shp_path)
        urllib.request.urlretrieve(SHX_URL, shp_path.replace('.shp', '.shx'))
        urllib.request.urlretrieve(DBF_URL, shp_path.replace('.shp', '.dbf'))
        print("Shapefile downloaded successfully!")
    except Exception as e:
        print(f"Could not download shapefile automatically: {e}")
        print("Fallback to coastline mode. Please place an official India .shp file in 'india_maps/'.")

# =====================================================================
# 2. Connect to Arraylake Data
# =====================================================================
print("Connecting to Arraylake...")
client = Client()
repo = client.get_repo("artamta/s2s-research")
session = repo.readonly_session("main")

try:
    ds = xr.open_zarr(session.store, group="mean_stddev")
except Exception:
    ds = xr.open_zarr(session.store) 

temp_var = [v for v in ds.data_vars if '2m' in v or 'air_temperature' in v and 'stddev' not in v and 'max' not in v and 'min' not in v][0]
precip_var = [v for v in ds.data_vars if 'precip' in v or 'tp' in v or 'total_precipitation' in v][0]

lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

lat_slice = slice(40.0, 0.0) if ds[lat_name][0] > ds[lat_name][-1] else slice(0.0, 40.0)
lon_slice = slice(60.0, 100.0)
ds_india = ds.sel({lat_name: lat_slice, lon_name: lon_slice})

# =====================================================================
# 3. Core Mapping Engine with Official Border Overlay
# =====================================================================
def plot_forecast_week(var_name, step_idx, cmap, label_suffix):
    data_var = ds_india[var_name]
    
    selector = {}
    if 'step' in data_var.dims:
        selector['step'] = step_idx
    if 'reference_time' in data_var.dims:
        selector['reference_time'] = 0

    slice_2d = data_var.isel(selector).compute()
    
    fig = plt.figure(figsize=(9, 8), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Solid background layers
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#caf0f8', zorder=1)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#f8f9fa', zorder=0)
    
    # Plot forecast data raster
    im = slice_2d.plot(
        ax=ax, transform=ccrs.PlateCarree(), 
        cmap=cmap, add_colorbar=False, robust=True, zorder=2
    )
    
    # -----------------------------------------------------------------
    # THE CRITICAL FIX: Add Official Border Overlay from Shapefile
    # -----------------------------------------------------------------
    if os.path.exists(shp_path):
        # Adds the full domestic map including POK and Aksai Chin safely over the data
        ax.add_geometries(
            Reader(shp_path).geometries(), 
            ccrs.PlateCarree(), 
            facecolor='none', 
            edgecolor='black', 
            linewidth=1.5, 
            zorder=4
        )
    else:
        # Fallback to general coastlines if file write fails on server profile
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=3)
    
    # Subtle dashed indicators for neighboring nations only (keeps layout clean)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.6, edgecolor='#95a5a6', linestyle='--', zorder=3)
    
    ax.set_extent([65.0, 98.0, 5.0, 38.0], crs=ccrs.PlateCarree())
    
    # Labels and cosmetics
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels, gl.right_labels = False, False
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    
    unit_str = data_var.attrs.get('units', 'N/A')
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.07, shrink=0.85, aspect=30)
    cbar.set_label(f"{var_name} [{unit_str}]", fontsize=11, fontweight='bold', labelpad=6)
    
    plt.title(f"Official Indian Grid: Week {step_idx + 1}\n{var_name} ({label_suffix})", fontsize=12, fontweight='bold', pad=12)
    
    filename = f"{OUTPUT_DIR}/official_india_W{step_idx + 1}_{var_name}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f" -> Saved: {filename}")

# Run pipeline
print("\nStarting compliant forecast visualization pipeline...")
for week_index in range(4):
    print(f"\n--- Week {week_index + 1} ---")
    plot_forecast_week(temp_var, week_index, 'RdYlBu_r', '2m Temperature')
    plot_forecast_week(precip_var, week_index, 'YlGnBu', 'Precipitation')

print(f"\nPipeline finished! View the figures inside the '{OUTPUT_DIR}' directory.")
