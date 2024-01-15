import xarray as xr
from tqdm import tqdm
import numpy as np

from os import makedirs
from os.path import join
import sys
sys.path.append("../../")

from config.config import Config

opt = Config( era5_path='../../settings/data.nc')
tibetan_coords = opt.tibetan_coords

standard_lon = tibetan_coords["lon"]
standard_lat = tibetan_coords["lat"]

missing_value = {"BD": -999, "SAND": -100, "SILT": -100, "CLAY": -100, "GRAV": -100}

raw_root = "../../data/raw/CSDL"

inference_root = "../../data/inference"
inference_root = join(inference_root, "gsde")
makedirs(inference_root, exist_ok=True)

for key in tqdm(missing_value):
    ds1 = xr.open_dataset(join(raw_root, f"{key}1.nc"))
    ds1 = ds1.interp(lon=standard_lon, lat=standard_lat)
    ds2 = xr.open_dataset(join(raw_root, f"{key}2.nc"))
    ds2 = ds2.interp(lon=standard_lon, lat=standard_lat)
    ds = xr.concat([ds1, ds2], dim="depth")
    values = ds[key].values
    values = np.where(values == missing_value[key], np.nan, values)
    ds[key].values = values
    ds = ds.mean(dim="depth", skipna=True)
    ds.to_netcdf(join(inference_root, f"{key}.nc"))
