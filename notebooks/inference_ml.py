import pickle
from os import makedirs
from os.path import join
import argparse

import pandas as pd
import xarray as xr
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    args = parser.parse_args()

    LAYER = args.layer
    OUTPUT_ROOT = args.output_root
    OUTPUT_ROOT = join(OUTPUT_ROOT, LAYER)
    makedirs(OUTPUT_ROOT, exist_ok=True)

    # load model
    with open(join(f"../checkpoints/AutoML_split_method_spatial_layer_{LAYER}_iid_adversial_validation_time_budget_600", "automl_total.pkl"), "rb") as f:
        automl = pickle.load(f)

    # load feature names
    with open(join(f"../checkpoints/AutoML_split_method_spatial_layer_{LAYER}_iid_adversial_validation_time_budget_360", "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    # load constant data
    DATA_ROOT = "../data/inference"
    dem_ds = xr.open_dataset(join(DATA_ROOT, "dem/dem.nc"))

    dem_df = dem_ds.to_dataframe().reset_index().rename(columns={"dem": "DEM"})

    tp_mean_ds = xr.open_dataset(join(DATA_ROOT, "era5_tp_mean.nc"))

    tp_mean_df = tp_mean_ds.to_dataframe().reset_index().rename(columns={"tp": "tp_mean"})

    GSDE_ROOT = join(DATA_ROOT, "gsde")
    bd_ds = xr.open_dataset(join(GSDE_ROOT, "BD.nc"))
    clay_ds = xr.open_dataset(join(GSDE_ROOT, "CLAY.nc"))
    grav_ds = xr.open_dataset(join(GSDE_ROOT, "GRAV.nc"))
    sand_ds = xr.open_dataset(join(GSDE_ROOT, "SAND.nc"))
    silt_ds = xr.open_dataset(join(GSDE_ROOT, "SILT.nc"))

    bd_df = bd_ds.to_dataframe().reset_index().rename(columns={"BD": "BD_mean"})
    clay_df = clay_ds.to_dataframe().reset_index().rename(columns={"CLAY": "CLAY_mean"})
    grav_df = grav_ds.to_dataframe().reset_index().rename(columns={"GRAV": "GRAV_mean"})
    sand_df = sand_ds.to_dataframe().reset_index().rename(columns={"SAND": "SAND_mean"})
    silt_df = silt_ds.to_dataframe().reset_index().rename(columns={"SILT": "SILT_mean"})

    l = [bd_df, clay_df, grav_df, sand_df, silt_df]
    gsde_df = l[0].copy()
    for item in l[1:]:
        gsde_df = pd.merge(gsde_df, item, on=["lat", "lon"])
    assert len(gsde_df) == len(bd_df)

    # load dynamic data
    era5_accumulations_ds = xr.open_dataset(join(join(DATA_ROOT, "era5_land_accumulations/accumulations.nc")))  # 1999-12-31 ~ 2022-12-30
    era5_tp_sum7_ds = xr.open_dataset(join(DATA_ROOT, "era5_tp_sum/tp_sum7.nc"))  # 2000-01-06 ~ 2022-12-30
    era5_tp_sum28_ds = xr.open_dataset(join(DATA_ROOT, "era5_tp_sum/tp_sum28.nc"))  # 2000-01-27 ~ 2022-12-30
    modis_lai_ds = xr.open_dataset(join(DATA_ROOT, "modis_lai/modis_lai.nc"))  # 2000-01-01 ~ 2021-12-27

    # gldas 2000-01-01 ~ 2021-12-31
    # era5_land_instantaneous 2000-01 ~ 2022-11

    # time 2000-01-27 ~ 2021-12-27

    years = list(range(2000, 2022))
    months = list(range(1, 13))

    for year in years:
        gldas_ds = xr.open_dataset(join(DATA_ROOT, f"gldas/{year}.nc"))

        for month in months:
            era5_instantaneous_ds = xr.open_dataset(join(DATA_ROOT, f"era5_land_instantaneous/{year}_{str(month).zfill(2)}.nc"))
            
            if month == 12:
                next_month = 1
                next_year = year + 1
            else:
                next_month = month + 1
                next_year = year

            times = np.arange(f"{year}-{str(month).zfill(2)}", f"{next_year}-{str(next_month).zfill(2)}", dtype='datetime64[D]')

            dataset_list = []

            for time in tqdm(times):
                era5_accumulations_df = era5_accumulations_ds.sel(time=time).to_dataframe().reset_index().rename(columns={"latitude": "lat", "longitude": "lon"}).drop("time", axis=1)
                era5_tp_sum7_df = era5_tp_sum7_ds.sel(time=time).to_dataframe().reset_index().rename(columns={"latitude": "lat", "longitude": "lon", "tp": "tp_sum7"}).drop("time", axis=1)
                era5_tp_sum28_df = era5_tp_sum28_ds.sel(time=time).to_dataframe().reset_index().rename(columns={"latitude": "lat", "longitude": "lon", "tp": "tp_sum28"}).drop("time", axis=1)
                modis_lai_df = modis_lai_ds.sel(time=time).to_dataframe().reset_index().drop("time", axis=1)

                gldas_df = gldas_ds.sel(time=time).to_dataframe().reset_index().drop("time", axis=1)
                era5_instantaneous_df = era5_instantaneous_ds.sel(time=time).to_dataframe().reset_index().rename(columns={"latitude": "lat", "longitude": "lon"}).drop("time", axis=1)

                l = [gsde_df, dem_df, tp_mean_df, era5_accumulations_df, era5_tp_sum7_df, era5_tp_sum28_df, modis_lai_df, gldas_df, era5_instantaneous_df]
                inference_df = l[0].copy()

                for item in l[1:]:
                    inference_df = pd.merge(inference_df, item, on=["lat", "lon"])
                assert len(inference_df) == len(era5_accumulations_df)

                for i in range(1, 13):
                    if i == month:
                        inference_df[f"month_{i}"] = 1
                    else:
                        inference_df[f"month_{i}"] = 0

                assert set(inference_df.columns.to_list()) == set(feature_names)

                inference_df = inference_df[feature_names]

                nan_index = inference_df[inference_df.isna().any(axis=1)].index

                inference_df_withnonan = inference_df.fillna(0)

                pred = automl.predict(inference_df_withnonan)

                inference_df_withnonan["pred"] = pred

                pred_df = inference_df_withnonan[["lat", "lon", "pred"]].copy()
                pred_df["time"] = time

                pred_df.loc[nan_index, "pred"] = np.nan
                pred_df = pred_df.rename(columns={"pred": "sm"})

                pred_ds = pred_df.set_index(["lat", "lon", "time"]).to_xarray()
                dataset_list.append(pred_ds)

            output_ds = xr.concat(dataset_list, dim="time")
            output_ds.to_netcdf(join(OUTPUT_ROOT, f"{year}_{str(month).zfill(2)}.nc"))
            