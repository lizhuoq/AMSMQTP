import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from os import listdir
from os.path import join
from datetime import datetime
import os


def get_nearest(arr, target):
    '''
    - output: index or None
    '''
    absolute_diff = np.abs(arr - target)
    resolution = np.abs(arr[1] - arr[0])
    if target >= np.min(arr) - resolution / 2 and target <= np.max(arr) + resolution / 2:
        return np.argmin(absolute_diff)
    else:
        return None 


def check_lai(raw_folder, insitu_path, daily_path, meta, region='Tibetan', show=True):
    '''
    - raw_folder: raw netcdf folder
    - insitu_path: gap-8days path  
    - daily_path: daily path
    '''
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    df = pd.read_csv(insitu_path)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.rename(columns={'lai': 'interpolated_lai'}, inplace=True)
    df.set_index('date_time', inplace=True)
    df.plot(ax=ax)

    df = pd.read_csv(daily_path)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.rename(columns={'lai': 'daily_lai'}, inplace=True)
    df.set_index('date_time', inplace=True)
    df.plot(ax=ax)

    idx = int(insitu_path.split('/')[-1].split('_')[0])
    lon = meta.loc[idx, [('longitude', 'val')]].values.item()
    lat = meta.loc[idx, [('latitude', 'val')]].values.item()

    years = range(2000, 2022, 1)
    laiLs = []
    for year in tqdm(years):
        ds = xr.open_dataset(join(raw_folder, f'lai_8-day_0.1_{year}.nc'))
        time = [datetime.strptime(f'{year}{t:03}', '%Y%j').date().strftime('%Y-%m-%d') for t in ds.time.data]
        if year == 2000:
            lon_idx = get_nearest(ds.lon.data, lon)
            lat_idx = get_nearest(ds.lat.data, lat)
        laiLs.append(
            pd.DataFrame(
                {
                    'date_time': time, 
                    'lai': ds['lai'][:, lat_idx, lon_idx].data
                }
            )
        )
    lai_df = pd.concat(laiLs)
    lai_df['date_time'] = pd.to_datetime(lai_df['date_time'])
    lai_df.rename(columns={'lai': 'raw_lai'}, inplace=True)
    lai_df.set_index('date_time', inplace=True)
    lai_df.plot(ax=ax)

    plt.tight_layout()
    plt.savefig(f'lai_{region}_{idx}.png')
    if show:
        plt.show()
    else:
        pass


def check_dem(idx, meta, dem_path, dem_table_path):
    ds = xr.open_dataset(dem_path)
    data = ds.dem.data
    lon = meta.loc[idx, [('longitude', 'val')]].values.item()
    lat = meta.loc[idx, [('latitude', 'val')]].values.item()
    ismn_dem_value = meta.loc[idx, [('elevation', 'val')]].values.item()
    lon_idx = get_nearest(ds.lon.data, lon)
    lat_idx = get_nearest(ds.lat.data, lat)
    dem_table = pd.read_csv(dem_table_path)
    dem_table_value = dem_table[dem_table['index'] == idx][['DEM']].values.item()
    print(f'raw dem value: {data[lat_idx, lon_idx].item()}')
    print(f'dem table value: {dem_table_value}')
    print(f'ismn dem value: {ismn_dem_value}')
    print(f'raw coords: lat: {lat} | lon: {lon}')
    print(f'grid coords: lat: {ds.lat.data[lat_idx]} | lon: {ds.lon.data[lon_idx]}')


def check_landcover(idx, meta, path, table_path):
    '''
    - path: raw netcdf path
    '''
    ds = xr.open_dataset(path)
    data = ds.LAND_COVER.data
    lon = meta.loc[idx, [('longitude', 'val')]].values.item()
    lat = meta.loc[idx, [('latitude', 'val')]].values.item()
    lon_idx = get_nearest(ds.lon.data, lon)
    lat_idx = get_nearest(ds.lat.data, lat)
    table = pd.read_csv(table_path)
    table_value = table[table['index'] == idx][['LAND_COVER']].values.item()
    print(f'raw value: {data[lat_idx, lon_idx]}')
    print(f'table value: {table_value}')
    print(f'raw coords: lat: {lat} | lon: {lon}')
    print(f'grid coords: lat: {ds.lat.data[lat_idx]} | lon: {ds.lon.data[lon_idx]}')


def check_csdl(idx, meta, folder, table_path, val_name, depth, coords=None):
    '''
    - folder: raw csdl folder  
    - depth : [1, 8]  
    - coords: dict of lon, lat Tibetan or Global  
    '''  
    depth_idx = 2 if depth // 5 else 1
    path = join(folder, val_name + str(depth_idx) + '.nc')
    ds = xr.open_dataset(path)
    if coords:
        ds = ds.interp(lon=coords['lon'], lat=coords['lat'])
    else:
        pass
    lon = meta.loc[idx, [('longitude', 'val')]].values.item()
    lat = meta.loc[idx, [('latitude', 'val')]].values.item()
    lon_idx = get_nearest(ds.lon.data, lon)
    lat_idx = get_nearest(ds.lat.data, lat)
    table = pd.read_csv(table_path)
    table_value = table[table['index'] == idx][[val_name + str(depth)]].values.item()
    data = ds[val_name][(depth % 4) - 1, lat_idx, lon_idx].data
    print(f'raw {val_name + str(depth_idx)}.nc depth: {(depth % 4) - 1} value: {data}')
    print(f'table {val_name + str(depth)} value: {table_value}')
    print(f'raw coords: lat: {lat} | lon: {lon}')
    print(f'grid coords: lat: {ds.lat.data[lat_idx]} | lon: {ds.lon.data[lon_idx]}')


def check_era5(insitu_path, raw_folder, meta, shifttime_path, val_name, accumulation: bool, region='Tibetan', show=True):
    idx = int(insitu_path.split('/')[-1].split('_')[0])
    lon = meta.loc[idx, [('longitude', 'val')]].values.item()
    lat = meta.loc[idx, [('latitude', 'val')]].values.item()
    val_type = 'accumulations' if accumulation else 'instantaneous'
    filenames = [join(raw_folder, x, 'data.nc') for x in listdir(raw_folder) if x.startswith(val_type)]
    from multiprocessing import Queue, Process, current_process


    def worker(inqueue, outqueue):
        for file in iter(inqueue.get, 'STOP'):
            ds = xr.open_dataset(file)
            lon_idx = get_nearest(ds.longitude.data, lon)
            lat_idx = get_nearest(ds.latitude.data, lat)
            outqueue.put(
                pd.DataFrame(
                    {
                        'date_time': ds.time.data, 
                        f'{val_name}': ds[val_name][:, lat_idx, lon_idx].data 
                    }
                )
            )
        outqueue.put(f'{current_process().name}: FINISH!')

        
    def manager():
        PROCESSES = os.cpu_count() - 1

        inqueue = Queue()
        outqueue = Queue()

        for i in range(PROCESSES):
            Process(target=worker, args=(inqueue, outqueue)).start()

        for file in filenames:
            inqueue.put(file)

        for i in range(PROCESSES):
            inqueue.put('STOP')

        stop_count = 0
        res_ls = []
        pbar = tqdm(total=len(filenames))
        while stop_count < PROCESSES:
            pbar.update()
            res = outqueue.get()
            if isinstance(res, str):
                if res[-7: ] == 'FINISH!':
                    stop_count += 1
            else:
                res_ls.append(res)

        fig, ax = plt.subplots(1, 1, figsize=(15, 4))

        df_raw: pd.DataFrame = pd.concat(res_ls)
        df_raw.sort_values('date_time', inplace=True)
        df_raw.set_index('date_time', inplace=True)
        df_raw.rename(columns={f'{val_name}': f'raw_{val_name}'}, inplace=True)
        df_raw.plot(ax=ax)

        df_insitu = pd.read_csv(insitu_path)
        df_insitu = df_insitu[['date_time', f'{val_name}']]
        df_insitu['date_time'] = pd.to_datetime(df_insitu['date_time'])
        df_insitu.set_index('date_time', inplace=True)
        df_insitu.rename(columns={f'{val_name}': f'insitu_{val_name}'}, inplace=True)
        df_insitu.plot(ax=ax)

        df_shifttime = pd.read_csv(shifttime_path)
        df_shifttime = df_shifttime[['date_time', f'{val_name}']]
        df_shifttime['date_time'] = pd.to_datetime(df_shifttime['date_time'])
        df_shifttime.set_index('date_time', inplace=True)
        df_shifttime.rename(columns={f'{val_name}': f'shifttime_{val_name}'}, inplace=True)
        df_shifttime.plot(ax=ax)

        plt.tight_layout()
        plt.savefig(f'ERA5_LAND_{region}_{val_name}_{idx}.png')

        if show:
            plt.show()
        else:
            pass

        pbar.close()
        inqueue.close()
        outqueue.close()


    manager()


def chcek_cumsum(date_time: str, origin_dataframe, cumsum_dataframe, period, val_name='tp'):
    origin_dataframe['date_time'] = pd.to_datetime(origin_dataframe['date_time'])
    cumsum_dataframe['date_time'] = pd.to_datetime(cumsum_dataframe['date_time'])
    date_time = pd.to_datetime(date_time)
    cumsum_val = cumsum_dataframe[cumsum_dataframe['date_time'] == date_time]
    assert cumsum_val.shape[0] == 1
    cumsum_val = cumsum_val[val_name + '_sum' + str(period)].values.item()

    res_df = origin_dataframe[
        (origin_dataframe['date_time'] > date_time - np.timedelta64(period, 'D')) &
        (origin_dataframe['date_time'] <= date_time)
    ]
    assert res_df.shape[0] == period
    res_val = res_df[val_name].values.sum().item()

    if round(res_val - cumsum_val) == 0:
        return True
    else:
        return False