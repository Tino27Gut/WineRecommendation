from datetime import datetime

import pandas as pd

from models.synthetic_user import SyntheticUserSimulator
from src.utils.utils import get_project_file_path

# Parámetros de la simulación
simulation_name = "simulation_10"
n_users = 2000
user_acq_start_date = datetime(2022,1,1)
user_acq_end_date = datetime(2024,1,1)
repurchases_allowed = True
avg_days_between_purchases = 30
std_days_between_purchases = 5
liked_churn_rate = 0.008
disliked_churn_rate = 0.65
max_repurchases = 20
verbose= True

# Saving Path
repurchase_text = "repchs" if repurchases_allowed else "single"
data_saving_path = get_project_file_path("src", "data", "synthetic", f"{simulation_name}.pkl")
params_saving_path = get_project_file_path("src", "data", "synthetic", f"{simulation_name}_params.pkl")

# Class Main Inputs
wines_df = pd.read_csv(get_project_file_path("src", "data", "transformed", "wines_clean.csv"))      # DataFrame de vinos
meals_df = pd.read_excel(get_project_file_path("src", "data", "raw", "meals", "Meals.xlsx"))        # DataFrame de comidas
pairings = pd.read_csv(get_project_file_path("src", "data", "processed", "aux", "pairings.csv"))    # DataFrames de pairings
pairings = list(pairings["pairings"])                                                               # Lista de pairings
taste_columns = ["body", "tannins", "sweetness", "acidity"]                                         # Columnas de sabor
grapes = pd.read_csv(get_project_file_path("src", "data", "processed", "aux", "grapes.csv"))        # DataFrame de uvas
grapes = list(grapes["grapes"])                                                                     # Lista de uvas

simulation_weights = {
    "rating": .25,
    "price_quality": .25,
    "rating_qty": .1,
    "user_similarity": .3,
    "main_pairing": .1
}

# Class Instantiation
user = SyntheticUserSimulator(
    wine_df=wines_df,
    meals_df=meals_df,
    pairing_cols=pairings,
    taste_cols=taste_columns,
    grape_cols=grapes,
    weights=simulation_weights
)

# Data Creation
data, params = user.generate_synthetic_data(
    n_users=n_users,
    first_start_date=user_acq_start_date,
    first_end_date=user_acq_end_date,
    repurchase=repurchases_allowed,
    avg_days_between_purchases=avg_days_between_purchases,
    std_days_between_purchases=std_days_between_purchases,
    liked_churn_rate=liked_churn_rate,
    disliked_churn_rate=disliked_churn_rate,
    max_repurchases=max_repurchases,
    verbose=verbose
)

pd.to_pickle(data, data_saving_path)
pd.to_pickle(params, params_saving_path)
print(f"Guardado en {data_saving_path}")
