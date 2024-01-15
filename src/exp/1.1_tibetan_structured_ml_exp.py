# %%
import pandas as pd
import numpy as np
import torch
from d2l import torch as d2l
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import os
import sys
sys.path.append('../../')
import argparse
import pickle
from joblib import dump
from typing import List
import random


def modify_data_distibutation(filenames, layer) -> List[pd.DataFrame]:
        layer_map = {
            'layer1': 'swvl1', 
            'layer2': 'swvl2', 
            'layer3': 'swvl3', 
            'layer4': 'swvl3', 
            'layer5': 'swvl3'
        }
        res = []
        for file in filenames:
            df = pd.read_csv(file)
            mean_val = df['soil_moisture'].mean()
            std_val = df['soil_moisture'].std()
            target_mean = df[layer_map[layer]].mean()
            target_std = df[layer_map[layer]].std()
            df['soil_moisture_origin'] = df['soil_moisture'].copy()
            df['soil_moisture'] = df['soil_moisture'].apply(lambda x: ((x - mean_val) / std_val) * target_std + target_mean)
            assert round(df['soil_moisture'].mean() - target_mean) == 0
            assert round(df['soil_moisture'].std() - target_std) == 0
            res.append(df)
        return res


def temporal_split(df_ls, test_size=0.2, val_size=0.1) -> List[List[pd.DataFrame]]:
    train, val, test = [], [], []
    for df in df_ls:
        train_len = int(df.shape[0] * (1 - (test_size + val_size)))
        val_len = int(df.shape[0] * val_size)
        train.append(df.iloc[: train_len, :])
        val.append(df.iloc[train_len: (train_len + val_len), :])
        test.append(df.iloc[(train_len + val_len):, :])
    return train, val, test


def del_features(original_features: List, delete_features: List) -> List:
    deled_featrues = []
    for o in original_features:
        for d in delete_features:
            if d in o or o.startswith(d):
                deled_featrues.append(o) 
            else:
                pass
    return list(set(original_features) - set(deled_featrues))


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



parser = argparse.ArgumentParser(description='Train Tibetan Soil Moisture Structured dataset using AutoML')
parser.add_argument('--layer', type=str, help='Soil Moisture Layer')
parser.add_argument('--tibetan_dataset_root', type=str, help='Tibetan Structured Dataset Root Path')
parser.add_argument('--cra_dataset_root', type=str, help='CRA Structured Dataset Root')
parser.add_argument('--split_method', type=str, choices=['spatial', 'temporal'], help='Split Method')
parser.add_argument('--log_root', type=str, default='../../logs', help='Log Root Path')
parser.add_argument('--checkpoints_root', type=str, default='../../checkpoints', help='Checkpoints Root Path')
parser.add_argument('--train', action='store_true', help='Train Model on Total Dataset')
parser.add_argument('--use_era5_mean_std', action='store_true', help='Use ERA5 Mean & Std adjust Observation Soil Moisture')
parser.add_argument('--del_features', type=str, default='Original', help='Features Deleted')
parser.add_argument('--train_seed', type=int, default=2023, help='AutoML Seed for Training')


if __name__ == '__main__':
    opt = parser.parse_args()

    tibetan_dataset_root = opt.tibetan_dataset_root
    cra_dataset_root = opt.cra_dataset_root
    layer = opt.layer
    split_method = opt.split_method
    SEED = random.randint(0, 10000)

    tibetan_filenames = load_tibetan_filenames(tibetan_dataset_root, layer)
    cra_filenames = load_cra_filenames(cra_dataset_root, layer)
    filenames = tibetan_filenames + cra_filenames

    # %%
    # spatial split  
    if split_method == 'spatial':
        train_filenames, test_filenames = train_test_split(filenames, test_size=0.2, shuffle=True, random_state=SEED)
        train_filenames, val_filenames = train_test_split(train_filenames, test_size=1 / 8, shuffle=True, random_state=SEED)
        if opt.use_era5_mean_std:
            train_data = pd.concat(modify_data_distibutation(train_filenames, layer), ignore_index=True)
            test_data = pd.concat(modify_data_distibutation(test_filenames, layer), ignore_index=True)
            val_data = pd.concat(modify_data_distibutation(val_filenames, layer), ignore_index=True)
        else:
            train_data = pd.concat([pd.read_csv(x) for x in train_filenames], ignore_index=True)
            test_data = pd.concat([pd.read_csv(x) for x in test_filenames], ignore_index=True)
            val_data = pd.concat([pd.read_csv(x) for x in val_filenames], ignore_index=True)
        all_features = pd.concat(
            (
                train_data.loc[:, [x for x in train_data.columns if x != 'soil_moisture']], 
                val_data.loc[:, [x for x in val_data.columns if x != 'soil_moisture']], 
                test_data.loc[:, [x for x in test_data.columns if x != 'soil_moisture']]
            )
        )
        all_features['month'] = pd.DatetimeIndex(all_features['date_time']).month
        all_features = all_features[[x for x in all_features.columns if x != 'date_time']]
        all_features = pd.get_dummies(all_features, columns=['LAND_COVER', 'month'], dtype=int)
        if opt.del_features == 'Original':
            pass
        else:
            all_features = all_features[del_features(all_features.columns, opt.del_features.split(','))]

    # %%
    # Temporal Split    
    if split_method == 'temporal':
        if opt.use_era5_mean_std:
            df_ls = modify_data_distibutation(filenames, layer)
        else:
            df_ls = [pd.read_csv(x) for x in filenames]
        train_ls, val_ls, test_ls = temporal_split(df_ls)
        train_data = pd.concat(train_ls, ignore_index=True)
        val_data = pd.concat(val_ls, ignore_index=True)
        test_data = pd.concat(test_ls, ignore_index=True)
        all_features = pd.concat(
            (
                train_data.loc[:, [x for x in train_data.columns if x != 'soil_moisture']], 
                val_data.loc[:, [x for x in val_data.columns if x != 'soil_moisture']], 
                test_data.loc[:, [x for x in test_data.columns if x != 'soil_moisture']]
            )
        )
        all_features['month'] = pd.DatetimeIndex(all_features['date_time']).month
        all_features = all_features[[x for x in all_features.columns if x != 'date_time']]
        all_features = pd.get_dummies(all_features, columns=['LAND_COVER', 'month'], dtype=int)
        if opt.del_features == 'Original':
            pass
        else:
            all_features = all_features[del_features(all_features.columns, opt.del_features.split(','))]

    print('features: ', all_features.columns)
    # %%
    # automl  
    from flaml import AutoML

    if split_method in ['spatial', 'temporal']:
        n_train = train_data.shape[0]
        n_val = val_data.shape[0]
        train_features = all_features[: n_train]
        val_features = all_features[n_train: (n_train + n_val)]
        test_features = all_features[(n_train + n_val): ]
        train_labels = train_data.soil_moisture.values.reshape(-1, 1)
        val_labels = val_data.soil_moisture.values.reshape(-1, 1)
        test_labels = test_data.soil_moisture.values.reshape(-1, 1)

        if opt.use_era5_mean_std:
            train_features.drop('soil_moisture_origin', axis=1, inplace=True)
            val_features.drop('soil_moisture_origin', axis=1, inplace=True)
            test_features.drop('soil_moisture_origin', axis=1, inplace=True)
        else:
            pass

        automl = AutoML()
        automl.fit(
            X_train=train_features, 
            y_train=train_labels, 
            metric='r2', 
            task='regression', 
            n_jobs=-1, 
            time_budget=90, 
            X_val=val_features, 
            y_val=val_labels, 
            seed=SEED
        )
    else:
        n_train = train_data.shape[0]
        train_features = all_features[: n_train].values
        test_features = all_features[n_train: ].values
        train_labels = train_data.soil_moisture.values.reshape(-1, 1)
        test_labels = test_data.soil_moisture.values.reshape(-1, 1)

        automl = AutoML()
        automl.fit(
            X_train=train_features, 
            y_train=train_labels, 
            metric='r2', 
            task='regression', 
            n_jobs=-1, 
            time_budget=90, 
            eval_method='cv', 
            seed=SEED
        )

    results = {}
    test_preds = automl.predict(test_features)
    results['r2'] = [r2_score(test_labels, test_preds)]
    results['mse'] = [mean_squared_error(test_labels, test_preds)]
    results['rmse'] = [mean_squared_error(test_labels, test_preds, squared=False)]
    results['ubrmse'] = [mean_squared_error(test_labels - test_labels.mean(), test_preds - test_preds.mean(), squared=False)]
    results['pearsonr'] = [np.corrcoef(test_preds.ravel(), test_labels.ravel())[0, 1].item()]
    results['mae'] = [mean_absolute_error(test_labels, test_preds)]
    results['split_method'] = [split_method]
    results['layer'] = [layer]
    results['use_era5_mean_std'] = [opt.use_era5_mean_std]
    results['seed'] = SEED
    if opt.use_era5_mean_std:
        results['origin_r2'] = [r2_score(test_data.soil_moisture_origin.values.reshape(-1, 1), test_preds)]
    df_results = pd.DataFrame(results)

    assert df_results.shape[0] == 1
    assert os.path.exists(opt.log_root)

    if opt.del_features == 'Original':
        log_path = os.path.join(opt.log_root, f'tibetan_{tibetan_dataset_root.split("/")[-1]}_cra_{cra_dataset_root.split("/")[-1]}' + '_original.csv')
    else:
        log_path = os.path.join(opt.log_root, f'tibetan_{tibetan_dataset_root.split("/")[-1]}_cra_{cra_dataset_root.split("/")[-1]}' + '_drop_' + opt.del_features.replace(',', '_') + '.csv')
    if not os.path.exists(log_path):
        df_results.to_csv(log_path, index=False)
    else:
        df = pd.read_csv(log_path)
        df = pd.concat((df, df_results))
        df.to_csv(log_path, index=False)

    # train in total dataset  
    # if opt.train:
    #     data = pd.concat([pd.read_csv(x) for x in filenames])

    #     features = data.copy().loc[:, [x for x in data.columns if x != 'soil_moisture']]
    #     labels = data.copy().soil_moisture.values.reshape(-1, 1)

    #     features['month'] = pd.DatetimeIndex(features['date_time']).month

    #     features = features[[x for x in features.columns if x not in ['date_time']]]

    #     numeric_features = features.select_dtypes(include=['float']).columns

    #     scaler = StandardScaler()

    #     scaler.fit(features[numeric_features])

    #     features[numeric_features] = scaler.transform(features[numeric_features])

    #     landcover_enc = OneHotEncoder(categories=[range(1, 25)], sparse_output=False)
    #     landcover_enc.fit(features[['LAND_COVER']])
    #     features[landcover_enc.get_feature_names_out()] = landcover_enc.transform(features[['LAND_COVER']])
    #     features = features[[x for x in features.columns if x != 'LAND_COVER']]

    #     time_enc = OneHotEncoder(categories=[range(1, 13)], sparse_output=False)
    #     time_enc.fit(features[['month']])
    #     features[time_enc.get_feature_names_out()] = time_enc.transform(features[['month']])
    #     features = features[[x for x in features.columns if x != 'month']]

    #     if opt.del_features == 'Original':
    #         pass
    #     else:
    #         features = features[del_features(features.columns, opt.del_features.split(','))]
    #     assert 'soil_moisture' not in scaler.feature_names_in_ 


    #     automl = AutoML()
    #     automl.fit(
    #         X_train=features, 
    #         y_train=labels, 
    #         metric='r2', 
    #         task='regression', 
    #         time_budget=90, 
    #         n_jobs=-1, 
    #         eval_method='cv', 
    #         seed=opt.train_seed
    #     )

    #     preds = automl.predict(features)
    #     print(f'final r2_score: {r2_score(labels, preds)}')

    #     if opt.del_features == 'Original':
    #         checkpoints_root = os.path.join(opt.checkpoints_root, dataset_root.split('/')[-1] + '_original', layer)
    #     else:
    #         checkpoints_root = os.path.join(opt.checkponts_root, dataset_root.split('/')[-1] + '_' + del_features.replace(',', '_'), layer)

    #     os.makedirs(checkpoints_root, exist_ok=True)

    #     with open(os.path.join(checkpoints_root, 'automl.pkl'), "wb") as f:
    #         pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    #     dump(scaler, os.path.join(checkpoints_root, 'standard_scaler.joblib'))
    #     dump(landcover_enc, os.path.join(checkpoints_root, 'landcover_encoder.joblib'))
    #     dump(time_enc, os.path.join(checkpoints_root, 'time_encoder.joblib'))