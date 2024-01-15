from ismn.interface import ISMN_Interface
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

import os
from os.path import join
from os import listdir
from typing import List

from .gridded_data import get_nearest

def get_ismn_metadata(path):
    ismn_data = ISMN_Interface(path)
    return ismn_data.metadata

def check_layers(path, meta, layers):
    idx = int(path.split('/')[-1].split('.')[0])
    layer = path.split('/')[-2]
    depth_from = meta.loc[idx, [('instrument', 'depth_from')]].values.item()
    depth_to = meta.loc[idx, [('instrument', 'depth_to')]].values.item()
    print(
        f'raw ismn data range from {depth_from} to {depth_to} | layer: {layer} range from {min(layers[layer])} to {max(layers[layer])}'
        )

def check_ismn_data(path, interim_path, region='Tibetan'):
    '''
    - path: path of raw ismn data .zip  
    - interim_path: daily ismn data path
    '''
    id = int(interim_path.split('/')[-1].split('.')[0])
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    ismn_data = ISMN_Interface(path)
    ts, meta = ismn_data.read(id, return_meta=True)

    ts.loc[ts['soil_moisture_flag'] != 'G', 'soil_moisture'] = np.nan
    ts.plot(ax=axes[0], title=f'Raw Data for ID {id}')
    
    ts_resample = ts[['soil_moisture']].resample('1D').mean()
    ts_resample.dropna(inplace=True)
    ts_resample.plot(ax=axes[1], title='Restore Data')

    df = pd.read_csv(interim_path)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.set_index('date_time', inplace=True)
    df.plot(ax=axes[2], title='Interim Data')

    plt.tight_layout()
    plt.savefig(f'{region}_{id}.png')
    plt.show()

    return ts, ts_resample, df

def cal_grid_averages(in_folder, meta, out_folder, lon, lat):
    os.makedirs(out_folder, exist_ok=True)
    min_time = pd.to_datetime('2000-01-01T00')
    max_time = pd.to_datetime('2022-12-31T23')
    idxs = [int(x.split('.')[0]) for x in listdir(in_folder)]
    coords = meta.loc[idxs, [('latitude', 'val'), ('longitude', 'val')]]
    coords['lon'] = coords[('longitude', 'val')].apply(lambda x: lon[get_nearest(lon, x)])
    coords['lat'] = coords[('latitude', 'val')].apply(lambda x: lat[get_nearest(lat, x)])
    for i, (_, groups) in enumerate(coords.groupby(['lat', 'lon'])):
        if groups.shape[0] == 1:
            df = pd.read_csv(join(in_folder, str(groups.index.values.item()) + '.csv'))
            df['date_time'] = pd.to_datetime(df['date_time'])
            df = df[(df['date_time'] <= max_time) & (df['date_time'] >= min_time)]
            df.to_csv(join(out_folder, str(groups.index.values.item()) + f'_group_{i}' + '.csv'), index=False)
        else:
            idxs = groups.index.values
            df_groups = pd.concat([pd.read_csv(join(in_folder, str(idx.item())) + '.csv') for idx in idxs])
            df_groups['date_time'] = pd.to_datetime(df_groups['date_time'])
            if df_groups.shape[0] == 0:
                df_groups.set_index('date_time', inplace=True)
            else:
                df_groups = df_groups.resample('1D', on='date_time').mean()
            df_groups.dropna(inplace=True)
            for idx in idxs:
                df_groups.to_csv(join(out_folder, str(idx.item()) + f'_group_{i}' + '.csv'))

def check_group(idxs: List[int], group: int, folder1, folder2):
    '''
    - folder1: daily_folder  
    - folder2: grid_averaged folder
    '''
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for idx in idxs:
        df = pd.read_csv(join(folder1, str(idx) + '.csv'))
        df['date_time'] = pd.to_datetime(df['date_time'])
        df.rename(columns={'soil_moisture': f'index: {idx}'}, inplace=True)
        df.set_index('date_time', inplace=True)
        df.plot(ax=ax)
    filenames = [join(folder2, x) for x in listdir(folder2) if x.endswith(f'group_{group}.csv')]
    group_df = pd.read_csv(filenames[0])
    group_df['date_time'] = pd.to_datetime(group_df['date_time'])
    group_df.rename(columns={'soil_moisture': 'averaged'}, inplace=True)
    group_df.set_index('date_time', inplace=True)
    group_df.plot(ax=ax)

    plt.tight_layout()
    plt.savefig(f'{group}.png')
    plt.show()