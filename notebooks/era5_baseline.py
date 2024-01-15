import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from flaml import AutoML

import os
import argparse
import sys
sys.path.append('../')

from config.config import Config


def load_tibetan_filenames(dataset_root, layer):
    groups = []
    filenames = []
    for file in os.listdir(os.path.join(dataset_root, layer)):
        if int(file.split('.')[0].split('_')[-1]) in groups:
            pass
        else:
            df = pd.read_csv(os.path.join(dataset_root, layer, file))
            if df.shape[0] == 0:
                pass
            else:
                groups.append(int(file.split('.')[0].split('_')[-1]))
                filenames.append(os.path.join(dataset_root, layer, file))
    return filenames


def load_cra_filenames(dataset_root, layer):
    groups = []
    filenames = []
    total_filenames = os.listdir(os.path.join(dataset_root, layer))
    if len(total_filenames) == 0:
        return []
    for file in total_filenames:
        if int(file.split('.')[0].split('_')[-1]) in groups:
            pass
        else:
            groups.append(int(file.split('.')[0].split('_')[-1]))
            filenames.append(os.path.join(dataset_root, layer, file))
    return filenames


def temporal_split(dataframes, test_size=0.2):
    train, test = [], []
    for df in tqdm(dataframes):
        n_train = int(len(df) * (1 - test_size))
        train.append(df.iloc[:n_train, :])
        test.append(df.iloc[n_train: , :])
    return train, test


def adversial_validation(train_dataframe, test_dataframe):
    df_train_adv = train_dataframe.copy()
    df_test_adv = test_dataframe.copy()
    df_train_adv['Is_test'] = 0
    df_test_adv['Is_test'] = 1
    df_adv = pd.concat([df_train_adv, df_test_adv], ignore_index=True)

    automl = AutoML()
    automl_settings = {
        'time_budget': 60, 
        'metric': 'accuracy', 
        'task': 'classification', 
        'eval_method': 'cv', 
        'n_jobs': -1
    }

    automl.fit(
        X_train=df_adv.drop('Is_test', axis=1), 
        y_train=df_adv['Is_test'], 
        **automl_settings
    )

    preds_adv = automl.predict_proba(df_adv.drop('Is_test', axis=1))[:, 1]

    return preds_adv


def final_train(X_train, y_train, X_test, y_test):
    final_train_X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    final_train_y = pd.concat([y_train, y_test], axis=0, ignore_index=True)
    automl = AutoML()
    automl_settings = {
    'metric': 'r2', 
    'task': 'regression', 
    'n_jobs': -1, 
    'time_budget': 200, 
    'seed': 2023
    }
    automl.fit(
        X_train=final_train_X, 
        y_train=final_train_y, 
        **automl_settings
    )
    return automl


def set_index(train_dfs, test_dfs):
    res = []
    n_train = len(train_dfs)
    for i, df_it in enumerate(train_dfs + test_dfs):
        df_it = df_it.copy()
        df_it['station_idx'] = i
        res.append(df_it)
    return res[: n_train], res[n_train: ]


def cal_metrics(df: pd.DataFrame):
    ubrmse = mean_squared_error(df['true'] - df['true'].mean(), df['pred'] - df['pred'].mean(), squared=False)
    rmse = mean_squared_error(df['true'], df['pred'], squared=False)
    r = np.corrcoef(df['true'], df['pred'])[0][1]
    return pd.Series(dict(rmse=rmse, r=r, ubrmse=ubrmse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tibetan_dataset_root', default='../data/processed/Tibetan/structured_dataset_v5/', type=str)
    parser.add_argument('--cra_dataset_root', default='../data/processed/CRA/structured_dataset/', type=str)
    parser.add_argument("--iid", type=str, choices=["random", "adversial_validation"])
    parser.add_argument('--layer', type=str)
    parser.add_argument('--split_method', type=str, choices=['spatial', 'temporal'])
    parser.add_argument('--logs_root', type=str)
    parser.add_argument('--test_results_root', type=str)

    args = parser.parse_args()
    tibetan_dataset_root = args.tibetan_dataset_root
    cra_dataset_root = args.cra_dataset_root
    iid = args.iid
    layer = args.layer
    split_method = args.split_method
    test_results_root = args.test_results_root
    logs_root = args.logs_root

    opt = Config()
    if iid == "random":
        seed = opt.tibetan_cra_seed[split_method][layer]
    elif iid == "adversial_validation":
        seed = opt.adversial_validation_seed[layer]

    root_setting = f"ERA5SM_split_method_{split_method}_layer_{layer}_iid_{iid}"
    test_results_root = os.path.join(test_results_root, root_setting)
    logs_root = os.path.join(logs_root, root_setting)

    tibetan_filenames = load_tibetan_filenames(tibetan_dataset_root, layer)
    cra_filenames = load_cra_filenames(cra_dataset_root, layer)

    filenames = tibetan_filenames + cra_filenames

    if split_method == 'spatial':
        train_filenames, test_filenames = train_test_split(filenames, test_size=0.2, shuffle=True, random_state=seed)
        train_ls = [pd.read_csv(x) for x in train_filenames]
        test_ls = [pd.read_csv(x) for x in test_filenames]
    else:
        df_ls = [pd.read_csv(x) for x in filenames]
        train_ls, test_ls = temporal_split(df_ls)

    train_ls, test_ls = set_index(train_ls, test_ls)
    train_data = pd.concat(train_ls, ignore_index=True)
    test_data = pd.concat(test_ls, ignore_index=True)

    # map
    layers_era5 = {
        "layer1": "swvl1", 
        "layer2": "swvl2", 
        "layer3": "swvl3", 
        "layer4": "swvl3", 
        "layer5": "swvl3"
    }
    era5_col = layers_era5[layer]

    idx = test_data[['station_idx', era5_col, "soil_moisture"]].copy()
    idx.rename(columns={"soil_moisture": "true", era5_col: "pred"}, inplace=True)
    os.makedirs(test_results_root, exist_ok=True)
    idx.to_csv(os.path.join(test_results_root, 'test_results.csv'), index=False)
    
    results = idx.groupby('station_idx').apply(cal_metrics).reset_index(drop=True)
    df_results = pd.DataFrame(results.mean()).T
    os.makedirs(logs_root, exist_ok=True)
    df_results.to_csv(os.path.join(logs_root, 'results.csv'), index=False)
