import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from tqdm import tqdm
from flaml import AutoML
from sklearn.model_selection import StratifiedKFold

import os
import argparse
import sys
import pickle
import shutil
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

    pred = automl.predict(df_adv.drop("Is_test", axis=1))
    true = df_adv["Is_test"]

    return accuracy_score(true, pred)


# correct adversarial validation
def correct_adversarial_validation(train_dataframe, test_dataframe):
    df_train_adv = train_dataframe.copy()
    df_test_adv = test_dataframe.copy()
    df_train_adv['Is_test'] = 0
    df_test_adv['Is_test'] = 1
    df_adv = pd.concat([df_train_adv, df_test_adv], ignore_index=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)

    X = df_adv.drop("Is_test", axis=1)
    y = df_adv["Is_test"]

    adv_scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        automl = AutoML()
        automl_settings = {
            'time_budget': 30, 
            'metric': 'accuracy', 
            'task': 'classification', 
            'eval_method': 'cv', 
            'n_jobs': -1
        }

        automl.fit(
            X_train=X_train, y_train=y_train, **automl_settings
        )

        y_pred = automl.predict(X_test)

        adv_scores.append(accuracy_score(y_test, y_pred))

    return sum(adv_scores) / len(adv_scores)


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
    parser.add_argument('--layer', type=str)
    parser.add_argument('--split_method', type=str, choices=['spatial'])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--distribution_root", type=str)

    args = parser.parse_args()
    tibetan_dataset_root = args.tibetan_dataset_root
    cra_dataset_root = args.cra_dataset_root
    layer = args.layer
    split_method = args.split_method
    seed = args.seed
    logs_root = args.distribution_root

    if not os.path.exists(logs_root):
        os.makedirs(logs_root)

    opt = Config()

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

    all_features = pd.concat(
    (
        train_data.loc[:, [x for x in train_data.columns if x != 'soil_moisture']], 
        test_data.loc[:, [x for x in test_data.columns if x != 'soil_moisture']]
    )
    )
    all_features['month'] = pd.DatetimeIndex(all_features['date_time']).month
    all_features.drop(['date_time', 'LAND_COVER'], axis=1, inplace=True)
    all_features = pd.get_dummies(all_features, columns=['month'], dtype=int)
    all_features.drop(['station_idx'], axis=1, inplace=True)

    n_train = train_data.shape[0]
    train_features = all_features[: n_train]
    test_features = all_features[n_train: ]
    train_labels = train_data['soil_moisture']
    test_labels = test_data['soil_moisture']

    # acc = adversial_validation(train_features, test_features)
    # correct adversarial validation
    acc = correct_adversarial_validation(train_features, test_features)

    # correct adversarial validation
    logs_path = os.path.join(logs_root, "logs_correct.csv")
    logs = {"accuracy": [acc], "split_method": [split_method], "layer": [layer], "seed": [seed]}
    df = pd.DataFrame(logs)

    if not os.path.exists(logs_path):
        df_logs = df
    else:
        df_logs = pd.read_csv(logs_path)
        df_logs = pd.concat([df_logs, df], axis=0, ignore_index=True)

    df_logs.to_csv(logs_path, index=False)
