import xarray as xr
import numpy as np
from tqdm import tqdm

import os

layers_era5 = {
    "layer1": "swvl1", 
    "layer2": "swvl2", 
    "layer3": "swvl3", 
    "layer4": "swvl3", 
    "layer5": "swvl3"
}
months = range(1, 13)
years = range(2000, 2022)

era5_root = "../../data/inference/era5_land_instantaneous"
save_root = "../../data/compare/monthly/era5"

for layer, val in layers_era5.items():
    os.makedirs(os.path.join(save_root, layer), exist_ok=True)
    for month in tqdm(months):
        lst = []
        for year in years:
            ds = xr.open_dataset(os.path.join(era5_root, f"{year}_{str(month).zfill(2)}.nc"))
            ds = ds[[val]]
            lst.append(ds)
        ds_cat = xr.concat(lst, dim="time")
        ds_mean = ds_cat.mean(dim="time", skipna=True)

        ds_mean.to_netcdf(os.path.join(save_root, layer, f"{str(month).zfill(2)}.nc"))
