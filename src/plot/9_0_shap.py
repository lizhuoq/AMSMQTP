import pickle
import sys
sys.path.append("../../")
import os

import shap
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from notebooks.cra_tibetan_ml import load_cra_filenames, load_tibetan_filenames


save_root = "../../data/plot/shap"
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

    explainer = shap.Explainer(model, seed=2023)
    shap_values = explainer(X)

    plot_features = [x for x in feature_names if not x.startswith("month")]

    # beeswarm
    shap.plots.beeswarm(shap_values, show=False, max_display=len(plot_features))
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, f"{layer}_beeswarm.pdf"))

    plt.close()

    # bar
    shap.plots.bar(shap_values, show=False, max_display=len(plot_features))
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, f"{layer}_bar.pdf"))

    plt.close()
