import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np

import os


def cal_metrics(df: pd.DataFrame):
    ubrmse = mean_squared_error(df['true'] - df['true'].mean(), df['pred'] - df['pred'].mean(), squared=False)
    rmse = mean_squared_error(df['true'], df['pred'], squared=False)
    # count
    count = len(df)
    r = np.corrcoef(df['true'], df['pred'])[0][1]
    return pd.Series(dict(rmse=rmse, r=r, ubrmse=ubrmse, count=count))


# spatial temporal
## AutoML time_budget 600 iid adversial_validation layer{1..5}
## rf_baseline time_budget 600 iid adversial_validation layer{1..5}
## era5sm iid adversial_validation layer{1..5}
## GLDASSM iid adversial_validation layer{1..5}
## smci iid adversial_validation layer{2..5}
output_root = "../../data/plot/insitu_metrics"
os.makedirs(output_root, exist_ok=True)


for s in ["spatial", "temporal"]:
    for l in tqdm(range(1, 6)):
        
        # automl 
        automl_res = pd.read_csv(f"../../test_results/AutoML_split_method_{s}_layer_layer{l}_iid_adversial_validation_time_budget_600/test_results.csv")
        automl_res = automl_res.groupby('station_idx').apply(cal_metrics).reset_index(drop=False)

        automl_res.to_csv(os.path.join(output_root, f"automl_{s}_layer{l}.csv"), index=False)

        # rf
        rf_res = pd.read_csv(f"../../baseline/test_results/RF_split_method_{s}_layer_layer{l}_iid_adversial_validation_time_budget_600/test_results.csv")
        rf_res = rf_res.groupby('station_idx').apply(cal_metrics).reset_index(drop=False)

        rf_res.to_csv(os.path.join(output_root, f"rf_{s}_layer{l}.csv"), index=False)

        # era5
        era5_res = pd.read_csv(f"../../baseline/test_results/ERA5SM_split_method_{s}_layer_layer{l}_iid_adversial_validation/test_results.csv")
        era5_res = era5_res.groupby('station_idx').apply(cal_metrics).reset_index(drop=False)

        era5_res.to_csv(os.path.join(output_root, f"era5_{s}_layer{l}.csv"), index=False)

        # gldas
        gldas_res = pd.read_csv(f"../../baseline/test_results/GLDASSM_split_method_{s}_layer_layer{l}_iid_adversial_validation/test_results.csv")
        gldas_res = gldas_res.groupby('station_idx').apply(cal_metrics).reset_index(drop=False)

        gldas_res.to_csv(os.path.join(output_root, f"gldas_{s}_layer{l}.csv"), index=False)

        if l == 1:
            pass
        else:
            # smci
            smci_res = pd.read_csv(f"../../baseline/test_results/SMCI_split_method_{s}_layer_layer{l}_iid_adversial_validation/test_results.csv")
            
            # dropna
            smci_res.dropna(axis=0, how="any", inplace=True)

            smci_res = smci_res.groupby('station_idx').apply(cal_metrics).reset_index(drop=False)

            smci_res.to_csv(os.path.join(output_root, f"smci_{s}_layer{l}.csv"), index=False)
