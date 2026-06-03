import xarray as xr
import os

OUTDIR = "/storage/raj.ayush/ERA5_India"

ds = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr",
    storage_options={"token": "anon"},
)

vars_keep = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_column_water_vapour",
    "total_precipitation",
]

india = ds[vars_keep].sel(
    latitude=slice(40, 0),
    longitude=slice(65, 100),
    time=slice("2024-01-01", "2026-05-28")
)

print(india)

india = india.chunk({
    "time": 168,   # one week
})

india.to_zarr(
    os.path.join(OUTDIR, "era5_india_surface_2024_2026.zarr"),
    mode="w"
)

print("Finished")