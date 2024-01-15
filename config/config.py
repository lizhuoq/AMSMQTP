import os
from os.path import join
from os import listdir
import sys
sys.path.append('../')

from ismn.interface import ISMN_Interface
import xarray as xr
import numpy as np

from utils.ismn_data import get_ismn_metadata


class Config():
    def __init__(self, raw_ismn_tibetan_path=None, raw_ismn_global_path=None, 
                 era5_path=None) -> None:
        self.layers =  {
            'layer1': (0.0, 0.1), 
            'layer2': (0.1, 0.3), 
            'layer3': (0.3, 0.5), 
            'layer4': (0.5, 0.8), 
            'layer5': (0.8, 1.1)
        }
        self._raw_ismn_tibetan = 'data/raw/Data_separate_files_header_19500101_20230826_9562_asrG_20230826.zip' \
            if not raw_ismn_tibetan_path else raw_ismn_tibetan_path
        self._raw_ismn_global = 'data/raw/Data_separate_files_header_20000101_20201231_9562_Crun_20230723.zip' \
            if not raw_ismn_global_path else raw_ismn_global_path
        self._era5_path = 'settings/data.nc' \
            if not era5_path else era5_path
        
        self.era5_variables_map = {
            'accumulations': {
                'potential_evaporation': 'pev', 
                'total_evaporation': 'e', 
                'total_precipitation': 'tp'
            }, 
            'instantaneous': {
                '2m_temperature': '2t', 
                'leaf_area_index_high_vegetation': 'lai_hv', 
                'leaf_area_index_low_vegetation': 'lai_lv', 
                'volumetric_soil_water_layer_1': 'swvl1', 
                'volumetric_soil_water_layer_2': 'swvl2', 
                'volumetric_soil_water_layer_1': 'swvl3'
            }
        }

        self.valid_map = {
            'BD': {
                'missing_value': -999, 
                'scale_fator': 0.01
            }, 
            'CLAY': {
                'missing_value': -100
            }, 
            'GRAV': {
                'missing_value': -100
            }, 
            'SAND': {
                'missing_value': -100
            }, 
            'SILT': {
                'missing_value': -100
            }, 
            'LAND_COVER': {
                'valid_list': range(1, 25)
            }, 
            'MODIS_LAI': {
                'value_range': [0, 10], 
                'closed': 'left & right', 
            }, 
            'ERA5_LAND_lai_lv': {
                'min_val': 0
            }, 
            'ERA5_LAND_lai_hv': {
                'min_val': 0
            }, 
            'DEM': {
                'value_range': [-407, 8752], 
                'missing_value': 9999
            }
        }

        self.tibetan_cra_seed = {
            'spatial': {
                'layer1': 8126, 
                'layer2': 2390, 
                'layer3': 1078, 
                'layer4': 6457, 
                'layer5': 4894
            }, 
            'temporal': {
                'layer1': 2834, 
                'layer2': 94, 
                'layer3': 4516, 
                'layer4': 2676, 
                'layer5': 8673
            }
        }

        self.adversial_validation_seed = {
            "layer1": 286, 
            "layer2": 301, 
            "layer3": 318, 
            "layer4": 330, 
            "layer5": 321
        }

        self.correct_adversarial_validation_seed = {
            "layer1": 97, 
            "layer2": 335, 
            "layer3": 53, 
            "layer4": 184, 
            "layer5": 197
        }
        
    @property
    def global_coords(self):
        lat = np.linspace(-90, 90, num=1801)
        lon = np.linspace(0, 360, num=3600, endpoint=False)
        lon[lon >= 180] -= 360
        return {
            'lon': lon, 'lat': lat
        }

    @property
    def tibetan_coords(self):
        ds = xr.open_dataset(self._era5_path)
        return {
            'lat': ds.latitude.data, 
            'lon': ds.longitude.data
        }
    
    @property
    def ismn_tibetan_meta(self):
        return get_ismn_metadata(self._raw_ismn_tibetan)
    
    @property
    def ismn_global_meta(self):
        return get_ismn_metadata(self._raw_ismn_global)