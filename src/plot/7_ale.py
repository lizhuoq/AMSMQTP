import pickle
import os
import sys
sys.path.append("../../")

from alibi.explainers import ALE, plot_ale
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from notebooks.cra_tibetan_ml import load_tibetan_filenames, load_cra_filenames


save_root = "../../data/plot/ale"
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

    X = data[feature_names].to_numpy()
    y = data["soil_moisture"].to_numpy()

    plot_features = [x for x in feature_names if not x.startswith("month")]

    ale = ALE(automl.predict, feature_names=feature_names, target_names=["soil_moisture"])
    exp = ale.explain(X)

    for col in tqdm(plot_features):
        plot_ale(exp, features=[col])
        plt.tight_layout()
        plt.savefig(os.path.join(save_root, f"{layer}_{col}.pdf"))
