import pickle
import os
import sys
sys.path.append("../../")

from sklearn.inspection import permutation_importance
import pandas as pd
import shap
from tqdm import tqdm

from notebooks.cra_tibetan_ml import load_tibetan_filenames, load_cra_filenames

# new
# new1
save_root = "../../data/plot/insitu_pi_si_new1"

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

    model = automl.model.estimator

    lon_lat = data[["lon", "lat"]].drop_duplicates().reset_index(drop=True)

    num_insitu = len(lon_lat)

    # shap
    explainer = shap.Explainer(model, seed=2023)

    lst = []
    for i in tqdm(range(num_insitu)):
        insitu_lon = lon_lat.iloc[i]["lon"]
        insitu_lat = lon_lat.iloc[i]["lat"]
        insitu_data = data[(data["lon"] == insitu_lon) & (data["lat"] == insitu_lat)]
        insitu_X = insitu_data[feature_names]
        insitu_y = insitu_data["soil_moisture"]

        # permutation importance
        result = permutation_importance(model, insitu_X, insitu_y, n_jobs=-1, random_state=2023)

        importances_mean = result.importances_mean

        pi_variable = feature_names[importances_mean.argmax()]

        # new1
        while pi_variable in ["lon", "lat"] or pi_variable.startswith("month"):
            max_index = importances_mean.argmax()

            importances_mean[max_index] = importances_mean.min()

            new_max_index = importances_mean.argmax()

            pi_variable = feature_names[new_max_index]

        # new
        pi_value = importances_mean.max()

        # shap importance
        shap_values = explainer(insitu_X)

        shap_importances = shap_values.abs.mean(0).values

        si_variable = feature_names[shap_importances.argmax()]

        # new1
        while si_variable in ["lon", "lat"] or si_variable.startswith("month"):
            max_index = shap_importances.argmax()

            shap_importances[max_index] = shap_importances.min()

            new_max_index = shap_importances.argmax()

            si_variable = feature_names[new_max_index]

        # new
        si_value = shap_importances.max()

        lst.append(
            pd.DataFrame({
                "lon": [insitu_lon], 
                "lat": [insitu_lat], 
                "pi_variable": [pi_variable], 
                "si_variable": [si_variable], 
                "pi_value": [pi_value],   # new
                "si_value": [si_value],   # new
            })
        )

    df = pd.concat(lst, axis=0, ignore_index=True)
    df.to_csv(os.path.join(save_root, f"{layer}.csv"), index=False)
