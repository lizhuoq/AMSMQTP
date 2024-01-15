import xarray as xr
from tqdm import tqdm

from os import listdir, makedirs
from os.path import join
import sys
sys.path.append("../../")

from config.config import Config

opt = Config( era5_path='../../settings/data.nc')
tibetan_coords = opt.tibetan_coords

standard_lon = tibetan_coords["lon"]
standard_lat = tibetan_coords["lat"]

raw_root = "../../data/external/GLDAS_netcdf"

inference_root = "../../data/inference"
inference_root = join(inference_root, "gldas")
makedirs(inference_root, exist_ok=True)

years = list(range(2000, 2022))

for year in years:
    l = []
    filenames = [join(raw_root, x) for x in listdir(raw_root) if x.endswith(".nc4") and int(x.split(".")[-6][1:5]) == year]  
    for file in tqdm(filenames):
        try:
            ds = xr.open_dataset(file)
            ds = ds.interp(lon=standard_lon, lat=standard_lat)
            l.append(ds)
        except:
            print(f"{file} has error")

    ds_concated = xr.concat(l, dim="time")
    ds_concated = ds_concated.sortby("time", ascending=True)
    ds_concated = ds_concated.resample(time="1D", skipna=True, closed="left", label="left").mean(dim="time", skipna=True)

    scale_factor = {"SoilMoi0_10cm_inst": 1 / 100, "SoilMoi10_40cm_inst": 1 / 300, "SoilMoi40_100cm_inst": 1 / 600}

    for key in scale_factor:
        ds_concated[key].values = ds_concated[key].values * scale_factor[key]

    ds_concated.to_netcdf(join(inference_root, f"{year}.nc"))
