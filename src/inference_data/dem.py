import xarray as xr

from os import makedirs
from os.path import join
import sys
sys.path.append("../../")

from config.config import Config

opt = Config( era5_path='../../settings/data.nc')
tibetan_coords = opt.tibetan_coords

standard_lon = tibetan_coords["lon"]
standard_lat = tibetan_coords["lat"]

raw_path = "../../data/interim/dem_concat.nc"
inference_root = "../../data/inference"
inference_root = join(inference_root, "dem")
makedirs(inference_root, exist_ok=True)

ds = xr.open_dataset(raw_path)
ds = ds.interp(lon=standard_lon, lat=standard_lat)
ds.to_netcdf(join(inference_root, "dem.nc"))
