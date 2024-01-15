import xarray as xr
from tqdm import tqdm

import os
import sys
sys.path.append("../../")

from config.config import Config

opt = Config(
    raw_ismn_global_path='../../data/raw/Data_separate_files_header_20000101_20201231_9562_Crun_20230723.zip', 
    raw_ismn_tibetan_path='../../data/raw/Data_separate_files_header_19500101_20230826_9562_asrG_20230826.zip', 
    era5_path='../../settings/data.nc'
)
tibetan_coords = opt.tibetan_coords

smci_root = "../../data/baseline/unzip_smci_9km"
layer_map = {
    "layer2": ["10cm", "20cm"], 
    "layer3": ["30cm", "40cm"], 
    "layer4": ["50cm", "60cm", "70cm"], 
    "layer5": ["80cm", "90cm", "100cm"]
}
save_root = "../../data/compare/monthly/smci"
months = range(1, 13)

for layer, depths in layer_map.items():
    os.makedirs(os.path.join(save_root, layer), exist_ok=True)
    filenames = []
    for depth in depths:
        filenames += [os.path.join(smci_root, depth, x) for x in os.listdir(os.path.join(smci_root, depth))]
    
    for month in months:
        lst = []
        for file in filenames:
            year = file.split("_")[-2]
            ds = xr.open_dataset(file).sel(time=f"{year}-{str(month).zfill(2)}")
            
            # interp
            ds_interp = ds.interp(lon=tibetan_coords["lon"], lat=tibetan_coords["lat"])

            lst.append(ds_interp)

        ds_cat = xr.concat(lst, dim="time")
        ds_mean = ds_cat.mean(dim="time", skipna=True)

        ds_mean.to_netcdf(os.path.join(save_root, layer, f"{str(month).zfill(2)}.nc"))
