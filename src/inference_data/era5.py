from tqdm import tqdm
import xarray as xr
import numpy as np
from dateutil.relativedelta import relativedelta

from os import listdir, makedirs
from os.path import join
import sys
from datetime import datetime
sys.path.append("../../")

from config.config import Config

opt = Config( era5_path='../../settings/data.nc')
tibetan_coords = opt.tibetan_coords

standard_lon = tibetan_coords["lon"]
standard_lat = tibetan_coords["lat"]

raw_root = "../../data/raw/ERA5_LAND_TIBETAN_UNZIP"
inference_root = "../../data/inference"
inference_accumulations_root = join(inference_root, "era5_land_accumulations")
inference_instantaneous_root = join(inference_root, "era5_land_instantaneous")
inference_sum_root = join(inference_root, "era5_tp_sum")
makedirs(inference_sum_root, exist_ok=True)
makedirs(inference_accumulations_root, exist_ok=True)
makedirs(inference_instantaneous_root, exist_ok=True)
accumulations_filenames = [join(raw_root, x, "data.nc") for x in listdir(raw_root) if x.startswith("accumulations")]


l = []
for file in tqdm(accumulations_filenames):
    ds = xr.open_dataset(file)
    assert (ds["longitude"].values == standard_lon).all()
    assert (ds["latitude"].values == standard_lat).all()
    ds["time"] = ds["time"].values - np.timedelta64(1, "D")
    l.append(ds)

ds_concated = xr.concat(l, dim="time")
ds_concated = ds_concated.sortby("time", ascending=True)
ds_concated.to_netcdf(join(inference_accumulations_root, "accumulations.nc"))

time = ds_concated["time"].values
assert (time[-1] - time[0]).astype("timedelta64[D]") / np.timedelta64(1, "D") + 1 == len(time)

tp: xr.Dataset = ds_concated[["tp"]]
tp_sum7 = tp.rolling(dim={"time": 7}).sum(dim="time", skipna=True)

tp_sum28 = tp.rolling(dim={"time": 28}).sum(dim="time", skipna=True)

tp_sum28.to_netcdf(join(inference_sum_root, "tp_sum28.nc"))

tp_sum7.to_netcdf(join(inference_sum_root, "tp_sum7.nc"))

start_time = datetime(2000, 1, 1)
end_time = datetime(2022, 12, 1)

current_time = start_time
while current_time < end_time:
    current_year = current_time.year
    current_month = current_time.month
    next_time = current_time + relativedelta(months=1)
    next_year = next_time.year
    next_month = next_time.month
    current_ds = xr.open_dataset(join(raw_root, f"instantaneous_{current_year}_{str(current_month).zfill(2)}", "data.nc"))
    next_ds = xr.open_dataset(join(raw_root, f"instantaneous_{next_year}_{str(next_month).zfill(2)}", "data.nc"))
    ds_concated = xr.concat([current_ds, next_ds], dim="time")
    ds_concated = ds_concated.resample(time="D", skipna=True, closed="right", label="left").mean(dim="time", skipna=True)
    ds_concated = ds_concated.sel(time=f"{current_year}-{current_month}")
    ds_concated.to_netcdf(join(inference_instantaneous_root, f"{current_year}_{str(current_month).zfill(2)}.nc"))
    current_time = next_time
