import pickle
import os
import sys
sys.path.append("../../")

from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np

from notebooks.cra_tibetan_ml import load_tibetan_filenames, load_cra_filenames

save_root = "../../data/plot/permutation_importance"
os.makedirs(save_root, exist_ok=True)

for layer in [f"layer{i}" for i in range(1, 6)]:
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

    result = permutation_importance(model, X, y, n_jobs=-1, random_state=2023)

    importances_mean = result.importances_mean
    importances_std = result.importances_std
    importances = result.importances

    np.savez(os.path.join(save_root, f"{layer}.npz"), importances_mean=importances_mean, 
             importances_std=importances_std, importances=importances, feature_names=feature_names)
