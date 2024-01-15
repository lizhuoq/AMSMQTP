import xarray as xr
import numpy as np
from tqdm import tqdm

import os

layers_gldas = {
    "layer1": "SoilMoi0_10cm_inst", 
    "layer2": "SoilMoi10_40cm_inst", 
    "layer3": "SoilMoi10_40cm_inst", 
    "layer4": "SoilMoi40_100cm_inst", 
    "layer5": "SoilMoi40_100cm_inst"
}
months = range(1, 13)
years = range(2000, 2022)

gldas_root = "../../data/inference/gldas"
save_root = "../../data/compare/monthly/gldas"

for layer, val in layers_gldas.items():
    os.makedirs(os.path.join(save_root, layer), exist_ok=True)
    for month in tqdm(months):
        lst = []
        for year in years:
            ds = xr.open_dataset(os.path.join(gldas_root, f"{year}.nc")).sel(time=f"{year}-{str(month).zfill(2)}")
            ds = ds[[val]]
            lst.append(ds)
        ds_cat = xr.concat(lst, dim="time")
        ds_mean = ds_cat.mean(dim="time", skipna=True)

        ds_mean.to_netcdf(os.path.join(save_root, layer, f"{str(month).zfill(2)}.nc"))
