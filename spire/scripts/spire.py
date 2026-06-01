import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def get_indian_subset(dataset):
    """
    Safely subsets the dataset to the Indian region (5°N to 38°N, 65°E to 98°E).
    Handles both ascending and descending coordinate axes.
    """
    # Define bounds for the Indian subcontinent
    lat_min, lat_max = 5.0, 38.0
    lon_min, lon_max = 65.0, 98.0
    
    # Check if latitude is descending (like ERA5: 90 down to -90)
    if dataset.latitude[0] > dataset.latitude[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
        
    # Check longitude format (0 to 360 vs -180 to 180)
    if dataset.longitude.max() > 180:
        # Convert bounds to 0-360 if necessary
        lon_min_360 = lon_min if lon_min >= 0 else lon_min + 360
        lon_max_360 = lon_max if lon_max >= 0 else lon_max + 360
        lon_slice = slice(lon_min_360, lon_max_360)
    else:
        lon_slice = slice(lon_min, lon_max)
        
    return dataset.sel(latitude=lat_slice, longitude=lon_slice)


def print_data_diagnostics(ds_india):
    """
    Computes and prints coordinate ranges, time limits, 
    and min/max values for every variable in the subset.
    """
    print("="*60)
    print("        S2S HINDCAST DIAGNOSTICS (INDIAN REGION)        ")
    print("="*60)
    
    # 1. Spatial and Temporal Limits
    print(f"Latitude Range  : {ds_india.latitude.min().values:.2f}° to {ds_india.latitude.max().values:.2f}°")
    print(f"Longitude Range : {ds_india.longitude.min().values:.2f}° to {ds_india.longitude.max().values:.2f}°")
    
    # Check for time/lead_time coordinate names dynamically
    time_coord = [coord for coord in ['time', 'lead_time', 'dayofyear'] if coord in ds_india.coords]
    for tc in time_coord:
        print(f"Time Axis ({tc}): Min = {ds_india[tc].min().values}, Max = {ds_india[tc].max().values}")
    
    print("-"*60)
    print(f"{'Variable Name':<25} | {'Min Value':<12} | {'Max Value':<12}")
    print("-"*60)
    
    # 2. Min/Max for every data variable
    for var in ds_india.data_vars:
        # Call .compute() or .values safely on the scaled slice
        v_min = float(ds_india[var].min().compute())
        v_max = float(ds_india[var].max().compute())
        print(f"{var:<25} | {v_min:<12.4f} | {v_max:<12.4f}")
    print("="*60 + "\n")


def plot_research_map(ds_india, variable_name, time_index=0, cmap='RdYlBu_r'):
    """
    Generates a publication-grade map over India with precise political 
    boundaries, gridlines, and clear typographic layouts.
    """
    # Dynamically select the first available time dimension step
    time_dims = [dim for dim in ['time', 'lead_time', 'dayofyear'] if dim in ds_india[variable_name].dims]
    
    if time_dims:
        data_to_plot = ds_india[variable_name].isel({time_dims[0]: time_index}).compute()
        title_suffix = f"({time_dims[0]} step: {time_index})"
    else:
        data_to_plot = ds_india[variable_name].compute()
        title_suffix = ""

    # Set up matplotlib figure with a professional DPI for layout validation
    fig = plt.figure(figsize=(9, 8), dpi=150)
    
    # Define a standard PlateCarree projection for regional maps
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Plot spatial data using xarray's meshgrid interface
    im = data_to_plot.plot(
        ax=ax, 
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=False,  # Custom colorbar handler for finer control
        robust=True          # Outlier protection for color stretching
    )
    
    # --- Professional Map Features ---
    # Add high-resolution country borders and coastlines
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.0, edgecolor='black')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8, edgecolor='black', linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='#f9f9f9', zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor='#ffffff', edgecolor='gray', linewidth=0.5)
    
    # Add precise political gridlines and format tick coordinates
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    
    # Custom axes padding limits for regional framing
    ax.set_extent([65.0, 98.0, 5.0, 38.0], crs=ccrs.PlateCarree())

    # --- Publication Aesthetics ---
    # Append attributes dynamically if metadata exists in the dataset
    unit_str = data_to_plot.attrs.get('units', 'Units Not Found')
    long_name = data_to_plot.attrs.get('long_name', variable_name)
    
    # Generate balanced colorbar along the bottom margin
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.07, shrink=0.85, aspect=30)
    cbar.set_label(f"{long_name} [{unit_str}]", fontsize=11, fontweight='bold', labelpad=6)
    cbar.ax.tick_params(labelsize=9)

    plt.title(f"S2S Climatology over India\n{variable_name} {title_suffix}", fontsize=13, fontweight='bold', pad=12)
    
    # Save the output figure using publication formatting constraints
    output_filename = f"india_{variable_name}_step{time_index}.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Saved publication-grade plot as: {output_filename}")
    plt.show()


# =====================================================================
# Execution Framework
# =====================================================================
# (Assuming 'ds' is your opened xarray dataset variable from XetHub)
if 'ds' in locals() or 'ds' in globals():
    # 1. Crop to the Indian Region
    ds_india = get_indian_subset(ds)
    
    # 2. Run limits and Min/Max diagnostics
    print_data_diagnostics(ds_india)
    
    # 3. Generate high-quality plots for all available variables
    # (Loops through every variable inside the mean_stddev group automatically)
    for variable in ds_india.data_vars:
        # Choose colormap types systematically depending on mean vs variation characteristics
        color_scheme = 'viridis' if 'std' in variable.lower() else 'RdYlBu_r'
        
        try:
            plot_research_map(ds_india, variable_name=variable, time_index=0, cmap=color_scheme)
        except Exception as e:
            print(f"Could not map variable {variable}. Error: {e}")
else:
    print("Error: 'ds' variable not detected in your workspace environment. Please connect to your Zarr store session first.")