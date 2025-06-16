from pathlib import Path
import pandas as pd

from src.models.synthetic_user import SyntheticUserSimulator
from src.utils.utils import get_project_file_path

# Cantidad de simulaciones a generar
simulation_name = "simulation_01"
n_simulations = 1000
saving_path = get_project_file_path("src", "data", "synthetic", f"{simulation_name}_n{n_simulations}.pkl")

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

data = user.generate_synthetic_data(n_simulations)
pd.to_pickle(data, saving_path)
print(f"Guardado en {saving_path}")
