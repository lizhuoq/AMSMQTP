import xarray as xr
from tqdm import tqdm

from datetime import datetime
from os import makedirs, listdir
from os.path import join
import sys
sys.path.append("../../")

from config.config import Config

opt = Config( era5_path='../../settings/data.nc')
tibetan_coords = opt.tibetan_coords

standard_lon = tibetan_coords["lon"]
standard_lat = tibetan_coords["lat"]

inference_root = "../../data/inference"
inference_root = join(inference_root, "modis_lai")
makedirs(inference_root, exist_ok=True)

raw_root = "../../data/raw/MODIS_LAI"
years = list(range(2000, 2022))

l = []
for year in tqdm(years):
    file = join(raw_root, f"lai_8-day_0.1_{year}.nc")
    ds = xr.open_dataset(file)
    ds = ds.interp(lon=standard_lon, lat=standard_lat)
    ds["time"] = [datetime.strptime(str(year) + str(x), "%Y%j") for x in ds["time"].values]
    l.append(ds)

ds_concated = xr.concat(l, dim="time")
ds_concated = ds_concated.resample(time="1D", closed="left", label="left").interpolate()
