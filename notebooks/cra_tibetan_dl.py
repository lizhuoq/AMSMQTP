# %%
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState

import os
import argparse
import sys
sys.path.append('../')

from config.config import Config

# %%
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

# %%
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

# %%
def file2dataframes(filenames):
    dataframes = []
    for file in tqdm(filenames):
        df = pd.read_csv(file)
        df['date_time'] = pd.to_datetime(df['date_time'])
        assert df.isna().sum().sum() == 0
        df = df.set_index('date_time').resample('1D').asfreq().reset_index(drop=False)
        df.sort_values('date_time', ignore_index=True, inplace=True)
        df['Is_valid'] = 1
        invalid_index = df[df.isna().any(axis=1)].index
        df.loc[invalid_index, 'Is_valid'] = 0
        df.interpolate(inplace=True)
        assert df.isna().sum().sum() == 0
        dataframes.append(df)
    return dataframes

# %%
def temporal_split(dataframes, test_size=0.2):
    train, test = [], []
    for df in tqdm(dataframes):
        n_train = int(len(df) * (1 - test_size))
        train.append(df.iloc[:n_train, :])
        test.append(df.iloc[n_train: , :])
    return train, test

# %%
def set_index(train_dfs, test_dfs):
    res = []
    n_train = len(train_dfs)
    for i, df_it in enumerate(train_dfs + test_dfs):
        df_it = df_it.copy()
        df_it['station_idx'] = i
        res.append(df_it)
    return res[: n_train], res[n_train: ]

# %% 
def ubrmse_rmse_r_r2(df: pd.DataFrame):
    ubrmse = mean_squared_error(df['true'] - df['true'].mean(), df['pred'] - df['pred'].mean(), squared=False)
    rmse = mean_squared_error(df['true'], df['pred'], squared=False)
    r = np.corrcoef(df['true'], df['pred'])[0][1]
    r2 = r2_score(df['true'], df['pred'])
    return pd.Series(dict(r2=r2, rmse=rmse, r=r, ubrmse=ubrmse))

def visual(true, preds, name):
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--tibetan_dataset_root', default='../data/processed/Tibetan/structured_dataset_v5/', type=str)
parser.add_argument('--cra_dataset_root', default='../data/processed/CRA/structured_dataset/', type=str)
parser.add_argument('--layer', type=str)
parser.add_argument('--split_method', type=str, choices=['spatial', 'temporal'])
parser.add_argument('--logs_root', type=str)
parser.add_argument('--checkpoints_root', type=str)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--test_results_root', type=str)
parser.add_argument('--seq_len', type=int)

# %% 
args = parser.parse_args()
tibetan_dataset_root = args.tibetan_dataset_root
cra_dataset_root = args.cra_dataset_root
layer = args.layer
split_method = args.split_method
checkpoints_root = args.checkpoints_root
logs_root = args.logs_root
test_results_root = args.test_results_root
seq_len = args.seq_len

opt = Config()
seed = opt.tibetan_cra_seed[split_method][layer]


# %% 
root_setting = f"BiLSTM_hidden_size_{args.hidden_size}_num_layers_{args.num_layers}_split_method_{split_method}_layer_{layer}_seq_len_{seq_len}"
checkpoints_root = os.path.join(checkpoints_root, root_setting)
logs_root = os.path.join(logs_root, root_setting)
test_results_root = os.path.join(test_results_root, root_setting)
os.makedirs(test_results_root, exist_ok=True)

# %%
tibetan_filenames = load_tibetan_filenames(tibetan_dataset_root, layer)
cra_filenames = load_cra_filenames(cra_dataset_root, layer)
filenames = tibetan_filenames + cra_filenames

if split_method == 'spatial':
    train_filenames, test_filenames = train_test_split(filenames, test_size=0.2, shuffle=True, random_state=seed)
    train_dfs, test_dfs = file2dataframes(train_filenames), file2dataframes(test_filenames)
else:
    dfs = file2dataframes(filenames)
    train_dfs, test_dfs = temporal_split(dfs, test_size=0.2)

train_dfs, test_dfs = set_index(train_dfs, test_dfs)

# %%
data_features = pd.concat(train_dfs + test_dfs, axis=0, ignore_index=True)
data_features.drop(['Is_valid', 'soil_moisture', 'LAND_COVER', 'date_time', 'tp_sum28', 'tp_sum7', 'station_idx'], axis=1, inplace=True)

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(data_features)

data_features = scaler.transform(data_features)

# %%
time_features = pd.concat([item[['date_time', 'station_idx']] for item in train_dfs + test_dfs], axis=0, ignore_index=True)
time_features['MonthOfYear'] = time_features['date_time'].dt.month
time_features['MonthOfYear'] = time_features['MonthOfYear'].astype('category')
time_features = pd.get_dummies(time_features, columns=['MonthOfYear'], dtype=float)
 
# %%
valid_features = pd.concat([item[['Is_valid']] for item in train_dfs + test_dfs], axis=0, ignore_index=True)

# %%
from torch.utils.data import DataLoader, Dataset

# %%
class TSDataset(Dataset):
    def __init__(self, flag='train', seq_len=seq_len) -> None:
        super().__init__()
        assert flag in ['train', 'test']
        self.flag = flag
        self.seq_len = seq_len
        self.__read_data__()

    def __read_data__(self):
        train_borders = np.cumsum([0] + [len(item) for item in train_dfs])
        test_borders = np.cumsum([0] + [len(item) for item in test_dfs])
        test_borders += train_borders[-1]
        self.borders = train_borders if self.flag == 'train' else test_borders
        self.df = data_features
        self.tf = time_features.drop(['date_time', 'station_idx'], axis=1)
        self.index = time_features[['date_time', 'station_idx']]
        self.vf = valid_features
        self.labels = pd.concat([item[['soil_moisture']] for item in train_dfs + test_dfs], axis=0, ignore_index=True)
        stIdxes = []
        for i in range(len(self.borders) - 1):
            stIdxes += list(range(self.borders[i], self.borders[i + 1] - self.seq_len + 1))
        self.stIdxes = stIdxes

    def __getitem__(self, index):
        begin = self.stIdxes[index]
        end = begin + self.seq_len
        if self.flag == 'train':
            return torch.from_numpy(self.df.iloc[begin: end].values), \
                torch.from_numpy(self.tf.iloc[begin: end].values), \
                    torch.from_numpy(self.vf.iloc[begin: end].values), \
                    torch.from_numpy(self.labels.iloc[begin: end].values)
        else:
            return self.index.iloc[begin: end], torch.from_numpy(self.df.iloc[begin: end].values), \
                torch.from_numpy(self.tf.iloc[begin: end].values), \
                    torch.from_numpy(self.vf.iloc[begin: end].values), \
                    torch.from_numpy(self.labels.iloc[begin: end].values)

    def __len__(self):
        return self.borders[-1] - self.borders[0] - (len(self.borders) - 1) * (self.seq_len - 1)

# %%
train_set = TSDataset('train')
test_set = TSDataset('test')

# %%
train_iter = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)

# %%
class MaskedMSELoss(nn.MSELoss):
    def forward(self, pred, label, is_valid):
        self.reduction = 'none'
        unweighted_loss = super(MaskedMSELoss, self).forward(pred, label)
        weighted_loss = unweighted_loss * is_valid
        return weighted_loss.sum() / is_valid.sum()

# %%
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device('cpu')

# %%
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

# %%
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, bidirectional, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=bidirectional)
        D = 2 if bidirectional else 1
        self.projection = nn.Linear(D * hidden_size, output_size)

    def forward(self, df, tf):
        input = torch.cat([df, tf], axis=-1)
        return self.projection(self.lstm(input)[0])

# %%
def eval(model, device,):
    dataframes = []
    model.eval()

    with torch.no_grad():
        for i, (idx, df, tf, vf, y) in enumerate(test_set):
            batch_df = df.unsqueeze(0).float().to(device)
            batch_tf = tf.unsqueeze(0).float().to(device)
            batch_vf = vf.unsqueeze(0).float().to(device)
            batch_y = y.unsqueeze(0).float().to(device)

            outputs = model(batch_df, batch_tf).clip(0, 1)

            pred = outputs.squeeze().detach().cpu().numpy()
            true = batch_y.squeeze().detach().cpu().numpy()
            sample_weight = batch_vf.squeeze().detach().cpu().numpy()
            temp = idx.copy()
            temp['pred'] = pred
            temp['true'] = true
            temp['Is_valid'] = sample_weight
            dataframes.append(temp)

            if (sample_weight.sum() == len(sample_weight)) & (i % 20 == 0):
                visual(true, pred, os.path.join(test_results_root, str(i) + '.pdf'))
                
    dataframes = pd.concat(dataframes, ignore_index=True)
    dataframes = dataframes[dataframes['Is_valid'] == 1]
    total_loss = dataframes.groupby('station_idx').apply(ubrmse_rmse_r_r2).reset_index()
    total_loss['global_r2'] = r2_score(dataframes['true'], dataframes['pred'])
    total_loss['global_rmse'] = mean_squared_error(dataframes['true'], dataframes['pred'], squared=False)
    total_loss['global_r'] = np.corrcoef(dataframes['true'], dataframes['pred'])[0][1]
    total_loss['global_ubrmse'] = mean_squared_error(dataframes['true'] - dataframes['true'].mean(), dataframes['pred'] - dataframes['pred'].mean(), squared=False)
    total_loss = total_loss.mean()
    total_loss.drop('station_idx', inplace=True)

    model.train()

    return total_loss['ubrmse'], total_loss

# %%
def train(model, patience, learning_rate, train_epochs, device):
    model.to(device)
    path = checkpoints_root
    if not os.path.exists(checkpoints_root):
        os.makedirs(checkpoints_root)

    train_steps = len(train_iter)

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model_optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MaskedMSELoss()

    for epoch in range(train_epochs):
        train_loss = []

        model.train()

        for i, (batch_df, batch_tf, batch_vf, batch_y) in enumerate(train_iter):
            model_optim.zero_grad()

            batch_df = batch_df.float().to(device)
            batch_tf = batch_tf.float().to(device)
            batch_vf = batch_vf.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_df, batch_tf)

            loss = criterion(outputs, batch_y, batch_vf)

            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print('\titers: {0}, epoch: {1} | loss: {2:.7f}'.format(i + 1, epoch + 1, loss.item()))

            loss.backward()
            model_optim.step()
        
        train_loss = np.average(train_loss)
        test_loss, _ = eval(model, device)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, test_loss))
        early_stopping(test_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

    return model

# %%

EPOCHS = 15
BATCHSIZE = 32
N_TRAIN_EXAMPLES = BATCHSIZE * 300

def define_model(trial: optuna.trial.Trial):
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    num_layers = trial.suggest_int('n_layers', 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    bid = trial.suggest_categorical('bidirectional', [True, False])

    model = Model(input_size=34, hidden_size=hidden_size, dropout=dropout, bidirectional=bid, num_layers=num_layers, output_size=1)

    return model


def objective(trial):
    device = try_gpu()
    model = define_model(trial).to(device)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    for epoch in range(EPOCHS):
        model.train()

        for batch_idx, (batch_df, batch_tf, batch_vf, batch_y) in enumerate(train_iter):
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            batch_df = batch_df.float().to(device)
            batch_tf = batch_tf.float().to(device)
            batch_vf = batch_vf.float().to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            output = model(batch_df, batch_tf)
            loss = MaskedMSELoss()(output, batch_y, batch_vf)
            loss.backward()
            optimizer.step()

        test_loss, _ = eval(model, device)

        trial.report(test_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return test_loss

# # %% optuna 
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100, timeout=60 * 90)
# pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
# complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
# print("Study statistics: ")
# print("  Number of finished trials: ", len(study.trials))
# print("  Number of pruned trials: ", len(pruned_trials))
# print("  Number of complete trials: ", len(complete_trials))
# print("Best trial:")
# trial = study.best_trial
# print("  Value: ", trial.value)

# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))


# %%
# train
model = Model(input_size=34, hidden_size=args.hidden_size, dropout=0.1, bidirectional=True, num_layers=args.num_layers, output_size=1)
model = train(model, patience=3, learning_rate=1e-4, train_epochs=10, device=try_gpu())

# %%
os.makedirs(logs_root, exist_ok=True)
res_path = os.path.join(logs_root, 'results.csv')
_, df_logs = eval(model, try_gpu())
df_logs = pd.DataFrame(df_logs).T
df_logs.to_csv(res_path, index=False)