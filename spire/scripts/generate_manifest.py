import os
import zarr
import xarray as xr
import pandas as pd
from arraylake import Client

print("Connecting to Arraylake store...")
client = Client()
repo = client.get_repo("artamta/s2s-research")
session = repo.readonly_session("main")
store = session.store

# Open the root Zarr group to discover the complete tree hierarchy
root_group = zarr.open_group(store, mode='r')
all_groups = ['/'] + list(root_group.group_keys())

output_file = "dataset_manifest.txt"
print(f"Starting deep inspection across {len(all_groups)} group paths...")

with open(output_file, "w") as f:
    f.write("=====================================================================\n")
    f.write("                 SPIRE S2S DATASET: COMPLETE DEEP MANIFEST           \n")
    f.write("=====================================================================\n")
    f.write(f"Generated via automated pipeline\n\n")

    for g_name in all_groups:
        f.write(f"\n{'#'*70}\n")
        f.write(f" GROUP PATH: {g_name}\n")
        f.write(f"{'#'*70}\n")
        
        try:
            # Open group using xarray
            if g_name == '/':
                ds = xr.open_zarr(store)
            else:
                ds = xr.open_zarr(store, group=g_name)
                
            # --- 1. COORDINATES & DIMENSIONS AUDIT ---
            f.write("\n[1] DIMENSIONS & CORE AXES:\n")
            f.write(f"  * Total Dimensions: {dict(ds.dims)}\n")
            
            for coord_name in ds.coords:
                coord = ds[coord_name]
                f.write(f"  * Coordinate '{coord_name}': shape={coord.shape}, dtype={coord.dtype}\n")
                
                # Check for temporal bounds (start/end dates)
                if 'time' in coord_name.lower() or 'ref' in coord_name.lower():
                    try:
                        vals = pd.to_datetime(coord.values.flatten())
                        f.write(f"    - Temporal Range : {vals[0].strftime('%Y-%m-%d %H:%M')} TO {vals[-1].strftime('%Y-%m-%d %H:%M')}\n")
                        f.write(f"    - Step Frequency : Count={len(vals)}\n")
                    except Exception:
                        f.write(f"    - Values Range   : {coord.values[0]} to {coord.values[-1]}\n")
                
                # Check for forecast lead step horizons
                elif coord_name.lower() == 'step' or 'lead' in coord_name.lower():
                    f.write(f"    - Forecast Steps : Min={coord.values.min()}, Max={coord.values.max()} (Total Steps: {len(coord)})\n")
            
            # --- 2. DATA VARIABLES DEEP DIVE ---
            f.write("\n[2] DATA VARIABLES MANIFEST:\n")
            if not ds.data_vars:
                f.write("  (No data variables explicitly exposed in this group context)\n")
            else:
                for var_name in ds.data_vars:
                    v = ds[var_name]
                    f.write(f"  • Variable Name: '{var_name}'\n")
                    f.write(f"    - Grid Layout: Dimensions={v.dims} | Shape={v.shape}\n")
                    
                    # Extract attributes safely
                    desc = v.attrs.get('long_name', v.attrs.get('description', 'No description provided.'))
                    units = v.attrs.get('units', 'No units declared')
                    
                    f.write(f"    - Description: {desc}\n")
                    f.write(f"    - Units      : {units}\n")
                    
                    # Mini-stats scan to detect range/scale
                    f.write(f"    - Data Scale : (Attributes Metadata: {dict(v.attrs)})\n\n")
                    
        except Exception as e:
            f.write(f"  [ERROR] Failed to extract data layout for group '{g_name}': {str(e)}\n")

print(f"\nSuccess! The deep manifest file has been safely written to: {output_file}")
print("You can view it using 'cat dataset_manifest.txt' or 'less dataset_manifest.txt'")
