import os
import argparse
import random

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from d2l import torch as d2l
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser(description="Train Tibetan Soil Moisture Structured Dataset using MLP")
parser.add_argument('--layer', type=str, help='Soil Moisture Layer')
parser.add_argument('--dataset_root', type=str, help='Structured Dataset Root Path')
parser.add_argument('--split_method', type=str, choices=['spatial', 'temporal'], help='Split Method')
parser.add_argument('--log_root', type=str, help='Log Root Path')
parser.add_argument('--checkpoints_root', type=str, help='Checkpoints Root Path')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')


def get_filenames(dataset_root, layer):
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


if __name__ == "__main__":
    opt = parser.parse_args()
    LAYER = opt.layer
    DATASET_ROOT = opt.dataset_root
    SPILT_METHOD = opt.split_method
    LOG_ROOT = opt.log_root
    CHECKPOINTS_ROOT = opt.checkpoints_root
    BATCH_SIZE = opt.batch_size
    N_EPOCHS = opt.num_epochs
    SEED = random.randint(0, 10000)

    os.makedirs(os.path.join(CHECKPOINTS_ROOT, LAYER), exist_ok=True)
    os.makedirs(os.path.join(LOG_ROOT, LAYER), exist_ok=True)

    filenames = get_filenames(DATASET_ROOT, LAYER)
    train_filenames, test_filenames = train_test_split(filenames, test_size=0.2, random_state=SEED, shuffle=True)
    train_data = pd.concat([pd.read_csv(x) for x in train_filenames], ignore_index=True)
    test_data = pd.concat([pd.read_csv(x) for x in test_filenames], ignore_index=True)
    all_features = pd.concat((train_data.drop('soil_moisture', axis=1), test_data.drop('soil_moisture', axis=1)))
    all_features['month'] = pd.DatetimeIndex(all_features['date_time']).month
    categorical_features = ['month', 'LAND_COVER']
    numeric_features = [x for x in all_features.columns if x not in categorical_features]
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())
    all_features[categorical_features] = all_features[categorical_features].astype('category')
    all_features = pd.get_dummies(all_features, dummy_na=True, dtype=float)

    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[: n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train: ].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.soil_moisture.values.reshape(-1, 1), dtype=torch.float32)
    test_labels = torch.tensor(test_data.soil_moisture.values.reshape(-1, 1), dtype=torch.float32)

    loss = nn.MSELoss()
    in_features = train_features.shape[1]

    net = nn.Sequential(
        nn.Linear(in_features, 1024), 
        nn.ReLU(), 
        nn.Dropout(), 
        nn.Linear(1024, 256), 
        nn.ReLU(), 
        nn.Dropout(), 
        nn.Linear(256, 1) 
    )

    train_iter = d2l.load_array((train_features, train_labels), BATCH_SIZE)
    optimizer = torch.optim.Adam(net.parameters())

    results = {'loss': [], 'r2': [], 'pearsonr': [], 'rmse': [], 'ubrmse': []}
    for epoch in range(1, N_EPOCHS + 1):
        train_bar = tqdm(train_iter)
        running_results = {'batch_size': 0, 'loss': 0}

        net.train()
        for X, y in train_bar:
            running_results['batch_size'] += BATCH_SIZE
            net.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            running_results['loss'] += l.item() * BATCH_SIZE

            train_bar.set_description(desc='[%d/%d] Loss: %.4f' % (epoch, N_EPOCHS, running_results['loss'] / running_results['batch_size']))

        net.eval()

        with torch.no_grad():
            test_preds = net(test_features)
            r2 = r2_score(test_labels.reshape(-1), test_preds.reshape(-1))
            pearsonr = np.corrcoef(test_labels.numpy().reshape(-1), test_preds.numpy().reshape(-1))[0, 1].item()
            rmse = mean_squared_error(test_labels.reshape(-1), test_preds.reshape(-1), squared=False)
            ubrmse = mean_squared_error((test_labels - test_labels.mean()), (test_preds - test_preds.mean()), squared=False)

        results['loss'].append(running_results['loss'] / running_results['batch_size'])
        results['r2'].append(r2)
        results['pearsonr'].append(pearsonr)
        results['rmse'].append(rmse)
        results['ubrmse'].append(ubrmse)

        torch.save(net.state_dict(), os.path.join(CHECKPOINTS_ROOT, LAYER, 'epoch_%d.pth' % (epoch)))

        if epoch % 10 == 0 and epoch != 0:
            dataframe = pd.DataFrame(
                data={'Loss': results['loss'], 'R2': results['r2'], 'Pearsonr': results['pearsonr'], 'RMSE': results['rmse'], 'ubRMSE': results['ubrmse']}, 
                index=range(1, epoch + 1)
            )
            dataframe.to_csv(os.path.join(LOG_ROOT, LAYER, 'train_results.csv'), index_label='Epoch')


