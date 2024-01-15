import os
import pickle
import sys
sys.path.append("../../")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

from notebooks.cra_tibetan_ml import load_tibetan_filenames, load_cra_filenames

exclude_features_map = {
    "layer1": ['lai_lv', 'lai', 'swvl2', 'swvl1', 'swvl3', 'SILT_mean', 'SAND_mean'], 
    "layer2": ['swvl2', 'swvl1', 'swvl3', 'SoilMoi40_100cm_inst', 'SoilMoi10_40cm_inst'], 
    "layer3": ['swvl2', 'swvl1', 'swvl3', 'SoilMoi10_40cm_inst', 'SoilMoi40_100cm_inst'], 
    "layer4": ['swvl2', 'swvl1', 'swvl3', 'SoilMoi10_40cm_inst', 'SoilMoi40_100cm_inst', 'SoilMoi0_10cm_inst'], 
    "layer5": ['swvl2', 'swvl1', 'swvl3', 'SoilMoi10_40cm_inst', 'SoilMoi40_100cm_inst', 'SoilMoi0_10cm_inst']
}

save_root = "../../data/plot/pdp_ice"
os.makedirs(save_root, exist_ok=True)

for layer, exclude_features in exclude_features_map.items():
    with open(f"../../checkpoints/AutoML_split_method_spatial_layer_{layer}_iid_adversial_validation_time_budget_600/automl_total.pkl", "rb") as f:
        automl = pickle.load(f)

    with open(f"../../checkpoints/AutoML_split_method_spatial_layer_{layer}_iid_adversial_validation_time_budget_360/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    tibetan_filenames = load_tibetan_filenames("../../data/processed/Tibetan/structured_dataset_v5/", f"{layer}")
    cra_filenames = load_cra_filenames("../../data/processed/CRA/structured_dataset/", f"{layer}")

    filenames = tibetan_filenames + cra_filenames

    data = pd.concat([pd.read_csv(x) for x in filenames], axis=0, ignore_index=True)
    data["month"] = pd.DatetimeIndex(data["date_time"]).month
    data.drop(["date_time", "LAND_COVER"], axis=1, inplace=True)
    data = pd.get_dummies(data, columns=['month'], dtype=int)

    X = data[feature_names]
    y = data["soil_moisture"]

    model = automl.model.estimator
    include_features = [x for x in feature_names if x not in exclude_features]

    # pdp
    for col in include_features:
        PartialDependenceDisplay.from_estimator(model, X, [col], n_jobs=-1, verbose=100)
        plt.tight_layout()
        plt.savefig(os.path.join(save_root, f"{layer}_{col}_pdp.pdf"))

    # ice
    for col in include_features:
        PartialDependenceDisplay.from_estimator(model, X, [col], kind="both", centered=True, n_jobs=-1, verbose=100)
        plt.tight_layout()
        plt.savefig(os.path.join(save_root, f"{layer}_{col}_ice.pdf"))
