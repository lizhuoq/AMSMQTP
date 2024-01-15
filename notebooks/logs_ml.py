import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

import os

TEST_RESULTS_ROOT = "../test_results"
LOGS_ROOT = "../logs"

dirs = [x for x in os.listdir(TEST_RESULTS_ROOT) if os.path.isdir(os.path.join(TEST_RESULTS_ROOT, x))]
for dir in tqdm(dirs):
    df = pd.read_csv(os.path.join(TEST_RESULTS_ROOT, dir, "test_results.csv"))
    y_true = df["true"].values.reshape(-1)
    y_pred = df["pred"].values.reshape(-1)
    res = {"r2_total": [r2_score(y_true, y_pred)], "r_total": [np.corrcoef(y_true, y_pred)[0][1]], 
           "rmse_total": [mean_squared_error(y_true, y_pred, squared=False)], 
           "ubrmse_total": [mean_squared_error(y_true - y_true.mean(), y_pred - y_pred.mean(), squared=False)]}
    df_res = pd.DataFrame(res)
    df_log = pd.read_csv(os.path.join(LOGS_ROOT, dir, "results.csv"))
    df_log = pd.concat([df_log, df_res], axis=1)
    df_log.to_csv(os.path.join(LOGS_ROOT, dir, "results_total.csv"), index=False)
