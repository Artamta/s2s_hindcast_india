import xarray as xr
from arraylake import Client

client = Client()
repo = client.get_repo("artamta/s2s-research")
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, group="mean_stddev")

print("="*50)
print("             DATASET TIMELINE AUDIT             ")
print("="*50)
if 'reference_time' in ds.coords:
    ref_times = ds['reference_time'].values
    print(f"Total Forecast Initializations: {len(ref_times)}")
    print(f"Earliest Start Date          : {ref_times[0]}")
    print(f"Latest Start Date            : {ref_times[-1]}")
else:
    print("No explicit reference_time coordinate found.")

if 'step' in ds.coords:
    print(f"Forecast Lead Steps          : {ds['step'].values}")
print("="*50)
