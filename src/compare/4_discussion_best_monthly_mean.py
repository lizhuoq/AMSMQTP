import xarray as xr
from tqdm import tqdm

import os

layers = range(1, 6)
months = range(1, 13)
years = range(2000, 2022)

# discussion
pred_root = "../../data/discussion/output_best_filter"
save_root = "../../data/compare/monthly/discussion_best"

for l in layers:
    os.makedirs(os.path.join(save_root, f"layer{l}"), exist_ok=True)
    for month in tqdm(months):
        lst = []
        for year in years:
            ds = xr.open_dataset(os.path.join(pred_root, f"layer{l}", f"{year}_{str(month).zfill(2)}.nc"))
            ds = ds[["sm"]]
            lst.append(ds)
        ds_cat = xr.concat(lst, dim="time")
        ds_mean = ds_cat.mean(dim="time", skipna=True)

        ds_mean.to_netcdf(os.path.join(save_root, f"layer{l}", f"{str(month).zfill(2)}.nc"))
