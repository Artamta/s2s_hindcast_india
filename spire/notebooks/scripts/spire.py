import numpy as np
import xarray as xr
import matplotlib
# Use the 'Agg' backend so matplotlib can generate images on a headless GPU server without crashing
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from arraylake import Client  # Added the missing Arraylake import

def get_indian_subset(dataset):
    """
    Safely subsets the dataset to the Indian region (5°N to 38°N, 65°E to 98°E).
    """
    lat_min, lat_max = 5.0, 38.0
    lon_min, lon_max = 65.0, 98.0
    
    # Check if latitude is descending (like ERA5: 90 down to -90)
    if dataset.latitude[0] > dataset.latitude[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
        
    # Check longitude format (0 to 360 vs -180 to 180)
    if dataset.longitude.max() > 180:
        lon_min_360 = lon_min if lon_min >= 0 else lon_min + 360
        lon_max_360 = lon_max if lon_max >= 0 else lon_max + 360
        lon_slice = slice(lon_min_360, lon_max_360)
    else:
        lon_slice = slice(lon_min, lon_max)
        
    return dataset.sel(latitude=lat_slice, longitude=lon_slice)


def print_data_diagnostics(ds_india):
    """
    Computes and prints coordinate ranges, time limits, and min/max values.
    """
    print("\n" + "="*60)
    print("        S2S HINDCAST DIAGNOSTICS (INDIAN REGION)        ")
    print("="*60)
    
    print(f"Latitude Range  : {ds_india.latitude.min().values:.2f}° to {ds_india.latitude.max().values:.2f}°")
    print(f"Longitude Range : {ds_india.longitude.min().values:.2f}° to {ds_india.longitude.max().values:.2f}°")
    
    time_coord = [coord for coord in ['time', 'lead_time', 'dayofyear'] if coord in ds_india.coords]
    for tc in time_coord:
        print(f"Time Axis ({tc}): Min = {ds_india[tc].min().values}, Max = {ds_india[tc].max().values}")
    
    print("-"*60)
    print(f"{'Variable Name':<25} | {'Min Value':<12} | {'Max Value':<12}")
    print("-"*60)
    
    for var in ds_india.data_vars:
        v_min = float(ds_india[var].min().compute())
        v_max = float(ds_india[var].max().compute())
        print(f"{var:<25} | {v_min:<12.4f} | {v_max:<12.4f}")
    print("="*60 + "\n")


def plot_research_map(ds_india, variable_name, time_index=0, cmap='RdYlBu_r'):
    """
    Generates a publication-grade map over India and saves it to disk.
    """
    time_dims = [dim for dim in ['time', 'lead_time', 'dayofyear'] if dim in ds_india[variable_name].dims]
    
    if time_dims:
        data_to_plot = ds_india[variable_name].isel({time_dims[0]: time_index}).compute()
        title_suffix = f"({time_dims[0]} step: {time_index})"
    else:
        data_to_plot = ds_india[variable_name].compute()
        title_suffix = ""

    fig = plt.figure(figsize=(9, 8), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    im = data_to_plot.plot(
        ax=ax, 
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=False, 
        robust=True          
    )
    
    # Map features
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.0, edgecolor='black')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8, edgecolor='black', linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='#f9f9f9', zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor='#ffffff', edgecolor='gray', linewidth=0.5)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    ax.set_extent([65.0, 98.0, 5.0, 38.0], crs=ccrs.PlateCarree())

    unit_str = data_to_plot.attrs.get('units', 'N/A')
    long_name = data_to_plot.attrs.get('long_name', variable_name)
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.07, shrink=0.85, aspect=30)
    cbar.set_label(f"{long_name} [{unit_str}]", fontsize=11, fontweight='bold', labelpad=6)

    plt.title(f"S2S Climatology over India\n{variable_name} {title_suffix}", fontsize=13, fontweight='bold', pad=12)
    
    # Save file and close cleanly to avoid memory leaks on server
    output_filename = f"india_{variable_name}_step{time_index}.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f" -> Saved publication plot: {output_filename}")
    plt.close(fig)


# =====================================================================
# Main Execution (Connecting to Arraylake automatically)
# =====================================================================
if __name__ == "__main__":
    print("Connecting to Arraylake Session...")
    try:
        client = Client()
        repo = client.get_repo("artamta/s2s-research")
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, group="mean_stddev")
        print("Successfully connected to dataset!")
    except Exception as e:
        print(f"Authentication/Connection failed! Error details: {e}")
        print("Tip: Make sure you ran 'arraylake auth login' in your terminal first.")
        exit(1)

    # 1. Crop to Indian Region
    ds_india = get_indian_subset(ds)
    
    # 2. Run diagnostics
    print_data_diagnostics(ds_india)
    
    # 3. Generate and save plots for every variable automatically
    print("Generating figures...")
    for variable in ds_india.data_vars:
        color_scheme = 'viridis' if 'std' in variable.lower() else 'RdYlBu_r'
        try:
            plot_research_map(ds_india, variable_name=variable, time_index=0, cmap=color_scheme)
        except Exception as e:
            print(f"Skipping variable {variable}. Error: {e}")
            
    print("\nAll tasks finished! Check your current directory for the generated .png files.")