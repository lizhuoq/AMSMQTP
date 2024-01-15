import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm


def cal_total_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.dropna(axis=0, how="any")

    y_true = dataframe["true"].values.reshape(-1)
    y_pred = dataframe["pred"].values.reshape(-1)

    res = {"r2_total": [r2_score(y_true, y_pred)], "r_total": [np.corrcoef(y_true, y_pred)[0][1]], 
           "rmse_total": [mean_squared_error(y_true, y_pred, squared=False)], 
           "ubrmse_total": [mean_squared_error(y_true - y_true.mean(), y_pred - y_pred.mean(), squared=False)]}

    df_res = pd.DataFrame(res)
    return df_res


# spatial temporal
## AutoML time_budget 600 iid adversial_validation layer{1..5}
## rf_baseline time_budget 600 iid adversial_validation layer{1..5}
## era5sm iid adversial_validation layer{1..5}
## GLDASSM iid adversial_validation layer{1..5}
## smci iid adversial_validation layer{2..5}


lst = []
for s in ["spatial", "temporal"]:
    for l in tqdm(range(1, 6)):
        
        # automl 
        automl_log = pd.read_csv(f"../../logs/AutoML_split_method_{s}_layer_layer{l}_iid_adversial_validation_time_budget_600/results.csv")
        automl_res = pd.read_csv(f"../../test_results/AutoML_split_method_{s}_layer_layer{l}_iid_adversial_validation_time_budget_600/test_results.csv")

        lst.append(
            pd.concat(
                [
                    automl_log, 
                    cal_total_metrics(automl_res.copy()), 
                    pd.DataFrame({"exp": ["AutoML"], "layer": [f"layer{l}"], "split_method": [s]})
                ], 
                axis=1
            )
        )

        # rf
        rf_log = pd.read_csv(f"../../baseline/logs/RF_split_method_{s}_layer_layer{l}_iid_adversial_validation_time_budget_600/results.csv")
        rf_res = pd.read_csv(f"../../baseline/test_results/RF_split_method_{s}_layer_layer{l}_iid_adversial_validation_time_budget_600/test_results.csv")

        lst.append(
            pd.concat(
                [
                    rf_log, 
                    cal_total_metrics(rf_res.copy()), 
                    pd.DataFrame({"exp": ["RF"], "layer": [f"layer{l}"], "split_method": [s]})
                ], 
                axis=1
            )
        )

        # era5
        era5_log = pd.read_csv(f"../../baseline/logs/ERA5SM_split_method_{s}_layer_layer{l}_iid_adversial_validation/results.csv")
        era5_res = pd.read_csv(f"../../baseline/test_results/ERA5SM_split_method_{s}_layer_layer{l}_iid_adversial_validation/test_results.csv")

        lst.append(
            pd.concat(
                [
                    era5_log, 
                    cal_total_metrics(era5_res.copy()), 
                    pd.DataFrame({"exp": ["ERA5"], "layer": [f"layer{l}"], "split_method": [s]})
                ], 
                axis=1
            )
        )

        # gldas
        gldas_log = pd.read_csv(f"../../baseline/logs/GLDASSM_split_method_{s}_layer_layer{l}_iid_adversial_validation/results.csv")
        gldas_res = pd.read_csv(f"../../baseline/test_results/GLDASSM_split_method_{s}_layer_layer{l}_iid_adversial_validation/test_results.csv")

        lst.append(
            pd.concat(
                [
                    gldas_log, 
                    cal_total_metrics(gldas_res.copy()), 
                    pd.DataFrame({"exp": ["GLDAS"], "layer": [f"layer{l}"], "split_method": [s]})
                ], 
                axis=1
            )
        )

        if l == 1:
            pass
        else:
            # smci
            smci_log = pd.read_csv(f"../../baseline/logs/SMCI_split_method_{s}_layer_layer{l}_iid_adversial_validation/results.csv")
            smci_res = pd.read_csv(f"../../baseline/test_results/SMCI_split_method_{s}_layer_layer{l}_iid_adversial_validation/test_results.csv")

            lst.append(
                pd.concat(
                    [
                        smci_log, 
                        cal_total_metrics(smci_res.copy()), 
                        pd.DataFrame({"exp": ["SMCI"], "layer": [f"layer{l}"], "split_method": [s]})
                    ], 
                    axis=1
                )
            )

df_output = pd.concat(lst, axis=0, ignore_index=True)
df_output.to_csv("../../data/plot/metrics.csv", index=False)
