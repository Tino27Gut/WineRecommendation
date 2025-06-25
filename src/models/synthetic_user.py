import random
import logging
from typing import Dict, Tuple, List, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

import src.utils.utils as ut

default_grapes = pd.read_csv(ut.get_project_file_path("src", "data", "processed", "aux", "grapes.csv"))        # DataFrame de uvas
default_grapes = list(default_grapes["grapes"])    
default_pairings = pd.read_csv(ut.get_project_file_path("src", "data", "processed", "aux", "pairings.csv"))    # DataFrames de pairings
default_pairings = list(default_pairings["pairings"])
default_weights = {
    "rating": .25,
    "price_quality": .25,
    "rating_qty": .1,
    "user_similarity": .3,
    "main_pairing": .1
}

class SyntheticUserSimulator:
    """
    Esta clase simula usuarios que eligen vinos basados en preferencias sintéticas
    y evalúan si les gustó o no una recomendación.

    Objetivo: generar datasets sintéticos para entrenar los modelos de Machine Learning.
    """

    def __init__(self, wine_df: pd.DataFrame, meals_df: pd.DataFrame,
                 grape_cols: List[str]=default_grapes,
                 pairing_cols: List[str]=default_pairings,
                 taste_cols: List[str]=["body", "tannins", "sweetness", "acidity"],
                 weights: Dict[str, float]=default_weights,
                 top_n: int = 20):
        """
        Constructor: inicializa la clase con los datos base.

        Args:
            - wine_df -> DataFrame de vinos.
            - meals_df -> DataFrame de recetas/platos.
            - grape_cols -> Lista con nombre de las columnas a utilizar como uvas.
            - pairing_cols -> Lista con nombre de las columnas a utilizar como pairing.
            - taste_cols -> Lista con nombre de las columnas a utilizar como taste.
            - weights -> Diccionario con pesos para cada componente del score (rating, price_quality, rating_qty, user_similarity, main_pairing).
            - top_n -> Cantidad máxima de vinos que el usuario considera para elegir uno.
        """
        self.meals_df = meals_df.copy()
        self.pairing_cols = pairing_cols
        self.taste_cols = taste_cols
        self.grape_cols = grape_cols
        self.weights = weights
        self.top_n = top_n
        self.mm_scaler = MinMaxScaler()
        self.robust_scaler = RobustScaler()
        self.wine_df = wine_df

        # Pre-transformación de df (escalado tastes + agrega "otras uvas")
        self.tra_df, self.taste_cols_scld = self._transform_df()

        # Perfiles de pairing (cuantiles y stds por pairing y taste)
        self._taste_profiles, self._taste_deviations = self._build_pairing_profile()

    def generate_user_input(self) -> Dict:
        """
        Genera un input sintético de usuario (gustos, pairing, uvas, etc.).

        Returns:
            - Diccionario con los parámetros preferidos por el usuario.
        """
        selected_meal = self._select_random_meal()
        selected_grapes = self._select_grapes()
        selected_price_range = self._select_price_range()
        selected_profile = self._select_taste_profile(selected_meal.get("main_pairing"))

        user_input = {
            "meal": selected_meal["meal"],
            "main_pairing": selected_meal["main_pairing"],
            "pairing_list": selected_meal["pairing_list"],
            "grape_list": selected_grapes,
            "precio_min": selected_price_range[0],
            "precio_max": selected_price_range[1],
            "tastes": {
                "body": selected_profile["body_scld"][1],
                "tannins": selected_profile["tannins_scld"][1],
                "sweetness": selected_profile["sweetness_scld"][1],
                "acidity": selected_profile["acidity_scld"][1]
            },
            "tastes_label": {
                "body": selected_profile["body_scld"][0],
                "tannins": selected_profile["tannins_scld"][0],
                "sweetness": selected_profile["sweetness_scld"][0],
                "acidity": selected_profile["acidity_scld"][0]
            },
            "weights": self.weights
        }

        return user_input


    def _build_pairing_profile(self, quantiles=[0, .25, .5, .75, 1]) -> Tuple[Dict, Dict]:
        """
        Crea los perfiles de sabor por cuantiles por pairing.
            - Objetivo -> que haga sentido el perfil elegido con el seleccionado por el user.

        Args:
            - quantiles -> Lista con cuantiles a considerar.

        Returns:
            - profile -> Diccionario con cuantiles por sabor y pairing.
            - deviations -> Diccionario con desvíos estándar por sabor y pairing.
        """
        profiles = {}
        deviations = {}
        for pairing in self.pairing_cols:
            pairing_subset = self.tra_df[self.tra_df[pairing] == 1]
            profiles[pairing] = {}
            deviations[pairing] = {}
            for taste in self.taste_cols_scld:
                # Cuantiles
                taste_quantile = pairing_subset[taste].quantile(quantiles)
                profiles[pairing][taste] = taste_quantile

                # Desvío Estandar (para distribución normal en select_taste_profile)
                std = pairing_subset[taste].std()
                deviations[pairing][taste] = std

        return profiles, deviations
    

    def _select_random_meal(self) -> Dict[str, object]:
        """
        Selecciona una comida al azar y devuelve info útil para construir el perfil del usuario.

        Returns:
        - Dict con:
            - complete_meal (pd.Series) -> la comida completa (ingredientes, columnas one-hot).
            - meal (str) -> el nombre de la comida.
            - main_pairing (str) -> ingrediente principal en lowercase.
            - pairing_list (List[str]) -> lista de pairings que tienen valor 1.
        """
        i = np.random.randint(0, len(self.meals_df))
        complete_meal = self.meals_df.loc[i]

        meal = complete_meal["Comida"]

        main_pairing = complete_meal["Ingrediente Principal"].lower()

        # Excluye las 2 primeras columnas: "Comida" e "Ingrediente Principal"
        one_hot = complete_meal[2:]
        pairing_list = list(one_hot.index[one_hot > 0])

        return {
            "complete_meal": complete_meal,
            "meal": meal,
            "main_pairing": main_pairing,
            "pairing_list": pairing_list
        }
    
    
    def _select_taste_profile(self, main_pairing: str, prob: float = 0.2,
                          categories: List[str] = ["leve", "moderado", "marcado", "intenso"]) -> Dict[str, str]:
        """
        Selecciona un perfil de gustos de vino en base al pairing principal.
        Usa lógica probabilística con distribución normal y categorías definidas (sentido común + random).

        Args:
            - main_pairing -> pairing principal del plato elegido por el usuario.
            - prob -> probabilidad de elección de un sabor aleatorio.
            - categories -> categorías de las cuales puede seleccionar el usuario.

        Returns:
        - Diccionario con el gusto preferido por cada variable de sabor (body, sweetness, etc.)
        """
        selected_profile = {}
        for taste in self.taste_cols_scld:
            quantiles = self._taste_profiles[main_pairing][taste]
            std = self._taste_deviations[main_pairing][taste]

            if random.random() < prob:
                # Opción aleatoria entre cuantiles medios
                quant_random_options = [
                    (quantiles.values[i] + quantiles.values[i+1])/2
                    for i in range(len(quantiles.values)-1)
                ]
                selected_val = np.random.choice(quant_random_options)
            else:
                median = quantiles[.5]
                min_val = quantiles[0]
                max_val = quantiles[1]
                selected_val = min(max(np.random.normal(loc=median, scale=std), min_val), max_val)
            
            selected_profile[taste] = self._get_category(selected_val, quantiles, categories)
        return selected_profile


    def _get_category(self, value, quantile_values: pd.Series,
                      categories: List[str] = ["leve", "moderado", "marcado", "intenso"]) -> Tuple[str, List[float]]:
        """
        Clasifica un valor en una categoría sensorial (leve, moderado, etc.) según cuantiles.

        Args:
            - value -> valor a clasificar (escalado)
            - quantile_values -> serie de cuantiles para esa variable sensorial
            - categories -> etiquetas para cada rango de cuantiles

        Returns:
            - Tuple (categoría asignada, rango de valores del cuantil correspondiente)
        """
        if len(quantile_values)-1 != len(categories):
            raise IndexError("Length of quantiles ranges and categories does not match!")
        
        quant_values = quantile_values.values

        # Obtiene categoría y valor promedio del rango de los cuantiles
        for i in range(len(quant_values)-1):
            start_val = quant_values[i]
            end_val = quant_values[i+1]
            if start_val <= value < end_val:
                return categories[i], [start_val, end_val]
        
        # Gestiona valores atípicos (mayores que el máximo valor de los cuantiles)
        if value >= quant_values[-1]:
            return quant_values[-1], [quant_values[-2], quant_values[-1]]
        
        # Gestiona valores atípicos (menores que el mínimo valor de los cuantiles)
        if 0 <= value < quant_values[0]:
            return categories[0], [quant_values[0], quant_values[1]]
        
        # Error si el valor es inferior a cero
        raise ValueError(f"Passed value is less than zero! {value}")


    def _select_grapes(self,
                       top_n_grapes: int = 8,
                       min_grapes: int = 1,
                       max_grapes: int = 4) -> List[str]:
        """
        Selecciona aleatoriamente un set de uvas según su frecuencia en el dataset.

        Args:
            - top_n_grapes -> cuántas uvas populares considerar (resto = Otras Uvas).
            - min_grapes -> cantidad mínima de uvas a seleccionar.
            - max_grapes -> cantidad máxima de uvas a seleccionar.

        Returns:
            - Lista de uvas seleccionadas.
        """
        grapes_df = self.wine_df[self.grape_cols]
        top_grapes = self._get_top_grapes(grapes_df, top_n_grapes)
        total_grapes_in_df = sum(grapes_df.sum(axis=0))

        top_grapes_prob = top_grapes / total_grapes_in_df
        top_grapes_prob["Otras Uvas"] = 1 - sum(top_grapes_prob)
        grapes_prob = dict(top_grapes_prob)

        while True:
            grape_selection = [grape for grape, prob in grapes_prob.items() if random.random() < prob]
            if len(grape_selection) >= min_grapes:
                break
            
        return grape_selection[:max_grapes]
    

    def _get_top_grapes(self, grapes_df: pd.DataFrame, top_n_grapes: int) -> pd.Series:
        """
        Devuelve las top N uvas más frecuentes del dataset.

        Args:
            - grapes_df -> DataFrame con columnas de uvas (one-hot).
            - top_n_grapes -> número de uvas más frecuentes a devolver.

        Returns:
            - Serie con el total de apariciones por uva el el DataFrame (orden descendente).
        """
        top_grapes = grapes_df.sum(axis=0).sort_values(ascending=False).head(top_n_grapes)
        return top_grapes

    def _select_price_range(
            self,
            quantiles: List[float] = [.075, .125, .25, .375, .5, .625, .75, .875],
            low_prices_weights: List[float] = [.2, .3, .25, .15, .05, .025, .015, 0.01, 0, 0],
            min_dispersion: int = 2
    ) -> List[Union[int, None]]:
        """
        Selecciona un rango de precios realista con sesgo hacia precios bajos, simulando comportamiento de usuarios reales.

        Args:
            - quantiles -> cuantiles a considerar para definir posibles precios.
            - low_prices_weights -> pesos para elegir precios bajos (más probabilidad al principio).
            - min_dispersion -> distancia mínima entre el precio bajo y el alto en el rango de cuantiles.

        Returns:
            - Lista [min_price, max_price], donde max_price puede ser None si no hay límite superior.
        """
        df = self.wine_df

        # Eliminación de outliers (IQR * 1.5)
        no_outlier_price = ut.manage_outlier_IQR(df["price"], func="remove")
        price_quantiles = no_outlier_price.quantile(quantiles)

        # Adición de extremos (sin precio mínimo / máximo)
        price_quantiles[0], price_quantiles[1] = 0, -1
        price_quantiles = price_quantiles.sort_index()
        price_quantiles = pd.Series(price_quantiles.values)

        # Selección índice de cuantil de precio inferior
        low_price_index = random.choices(population=range(len(price_quantiles)), weights=low_prices_weights, k=1)[0]

        # Cálculo de límite superior posible
        high_top_limit = len(price_quantiles)
        high_bottom_limit = min(low_price_index + min_dispersion, high_top_limit-1)
        high_prices_possible = list(range(high_bottom_limit, high_top_limit))
        high_prices_weights = [1 / x**1.5 for x in high_prices_possible]

        # Selección de índice de cuantil de precio superior
        high_price_index = random.choices(population=high_prices_possible, weights=high_prices_weights, k=1)[0]

        # Rango de precios
        price_range = [int(price_quantiles[low_price_index]), int(price_quantiles[high_price_index])]
        if price_range[1] == -1:
            price_range[1] = None

        return price_range


    def _transform_df(self) -> Tuple[pd.DataFrame, List]:
        """
        Pre-transforma el DataFrame para la simulación.

        Returns:
        - DataFrame con:
            1. Tastes escalados (body, tannins, etc.)
            2. Columna "Otras Uvas".
            3. Métrica de calidad-precio global.
        - Lista con nombre de columnas de sabor escaladas.
        """
        tra_df = self.wine_df.copy()
        tra_df, taste_cols_scld = self._df_add_scaled_tastes(tra_df)  # Escala tastes
        tra_df = self._df_group_other_grapes(tra_df)                  # Agrupa Otras Uvas
        tra_df = self._df_add_qual_price(tra_df)                      # Agrega Métrica Calidad-Precio
        return tra_df, taste_cols_scld


    def _df_add_scaled_tastes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """
        Escala columnas de sabor del dataframe (body, tannins, etc.)

        Args:
            - df -> DataFrame con las columnas taste a escalar.

        Returns:
            - DataFrame con columnas de sabor Min-Max scaled (sufijo '_scld').
            - Lista con nombres de columnas de sabor escaladas.
        """
        tra_df = df.copy()
        taste_cols_scld = [taste + "_scld" for taste in self.taste_cols]
        tra_df[taste_cols_scld] = self.mm_scaler.fit_transform(tra_df[self.taste_cols])
        return tra_df, taste_cols_scld

    def _df_group_other_grapes(self, df: pd.DataFrame, top_n_grapes: int = 8, drop: bool = True) -> pd.DataFrame:
        """
        Agrega agrupa todas las uvas que no son 'top wines' en la columna 'Otras Uvas'.
        
        Args:
            - df -> el DataFrame con las columnas de uvas a agrupar.
            - top_n_grapes -> número de uvas más frecuentes no agrupadas en 'Otras Uvas'.
            - drop -> determina si dropea las columnas agrupadas en 'Otras Uvas' (default True).

        Returns:
            - Dataframe con uvas fuera del top_n_wines agrupadas en la columna 'Otras Uvas'.
        """
        tra_df = df.copy()
        grapes_df = tra_df[self.grape_cols]
        top_grapes = list(self._get_top_grapes(grapes_df, top_n_grapes).index)
        other_grapes = [grape for grape in self.grape_cols if grape not in top_grapes]
        tra_df["Otras Uvas"] = tra_df[other_grapes].max(axis=1)
        if drop:
            tra_df = tra_df.drop(columns=other_grapes)
        return tra_df

    
    # Se aplica 2 veces: 1) Global, 2) Local (por cada user)
    def _df_add_qual_price(self, df: pd.DataFrame) -> pd.DataFrame:
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
        """
        Agrega métricas de precio calidad al DataFrame.

        Args:
            df -> DataFrame al cual agregar métricas de precio calidad.
        
        Returns:
            - DataFrame con nuevas columnas:
                - rating_rscld -> rating con escalado robusto.
                - price_rscld -> precio con escalado robusto.
                - quality_price -> diferencia entre rating y precio escalados.
                - quality_price_rscld -> métrica de precio calidad entre 0 y 1 con escalado robusto.
        """
        tra_df = df.copy()
        # Cálculo de zscore para rating y price (mediana = 0, std = 1)
        rscaler = self.robust_scaler
        tra_df[['rating_rscld', 'price_rscld']] = rscaler.fit_transform(tra_df[["rating", "price"]])
                
        # Cálculo de métrica calidad-precio (calidad suma, precio resta)
        tra_df["quality_price"] = tra_df['rating_rscld'] - tra_df['price_rscld']
        
        # Escalado robusto con outliers fuera del fit, pero incluidos en el transform 
        # (Limita los outliers a IQR * 2, pone un límite de 0 y 1)
        no_outliers = ut.manage_outlier_IQR(
            tra_df['quality_price'], i=2, func="remove"
        )

        mod_mm_scaler = self.mm_scaler.fit(no_outliers.to_frame())

        # TODO: agregar visualización de distribución en el EDA

        # Límite de quality price en 0 y 1 (capear los outliers)
        scld_qual = np.clip(mod_mm_scaler.transform(tra_df[["quality_price"]]), a_min=0, a_max=1)
        tra_df["quality_price_rscld"] = scld_qual

        return tra_df


    def _score_wines(self, df: pd.DataFrame, user_input: Dict) -> pd.DataFrame:
        """
        Asigna un score a cada vino según el perfil del usuario.

        Args:
            - df -> DataFrame sobre el cual calcular el wine_score.
            - user_input -> Diccionario con gustos y preferencias del usuario.

        Returns:
            - DataFrame con vinos, sus scores y sus scores escalados.
        """
        tra_df = df.copy()

        # Cálculo de similitud de gustos del user contra cada vino
        user_tastes = user_input["tastes"]
        tra_df = self._df_add_user_similarity(tra_df, user_tastes)

        # Obtención de weights
        user_weights = user_input["weights"]
        rating_w = user_weights["rating"]
        price_quality_w = user_weights["price_quality"]
        rating_qty_w = user_weights["rating_qty"]
        user_similarity_w = user_weights["user_similarity"]
        main_pairing_w = user_weights["main_pairing"]

        # Puntaje sintético del vino
        tra_df["wine_score"] = (
            rating_w * (tra_df["rating"] / 5) + 
            price_quality_w * tra_df["quality_price_rscld"] + 
            rating_qty_w * (tra_df["rating_qty"] / tra_df["rating_qty"].max()) +
            user_similarity_w * tra_df["user_similarity"] + 
            main_pairing_w * tra_df[user_input["main_pairing"]]
        )

        # Min-Max Scaling del score
        tra_df["wine_score_scld"] = self.mm_scaler.fit_transform(tra_df[["wine_score"]])

        return tra_df


    def _df_add_user_similarity(self, df: pd.DataFrame, tastes: Dict) -> pd.DataFrame:
        """
        Agrega columna de similitud con el input del usuario al df.

        Args:
            - df -> Dataframe al que agregar la columna 'user_similarity'
            - tastes -> Diccionario con gustos y preferencias del usuario.

        Returns:
            - Dataframe con columna 'user_similarity' agregada.
        """

        tra_df = df.copy()
        # Calcula la similitud con los gustos del usuario con Fuzzy Distance
        distances = self._compute_fuzzy_distance(tra_df[self.taste_cols_scld], tastes)
        tra_df["user_similarity"] = 1 - (distances / distances.max())
        tra_df["taste_distance_raw"] = distances
        return tra_df
        

    def _compute_fuzzy_distance(self, tastes_df: pd.DataFrame, tastes: Dict) -> np.array:
        """
        Computa la distancia tipo fuzzy entre los gustos del usuario y los vinos disponibles.

        Args:
        - tastes_df -> DataFrame de columnas taste escaladas.
        - tastes -> Diccionario con gustos y preferencias del usuario.

        Returns:
        - Numpy Array con las distancias entre cada registro del DataFrame y los gustos del usuario.
        """
        # Cálculo de distancias por cada vino (registro)
        distances = []
        for _, row in tastes_df.iterrows():
            wine_distance = 0
            col_count = 0
            # Calculo de Fuzzy Distance de cada taste (columna)
            for col in tastes_df.columns:
                key = col.replace("_scld", "")
                if key in tastes:
                    value = row[col]
                    q_low, q_high = tastes[key]
                    wine_distance += self._fuzzy_distance(value, q_low, q_high)
                    col_count += 1
            
            # Cálculo del promedio de distancias
            distance = wine_distance / col_count if col_count > 0 else None
            distances.append(distance)

        distances = np.array(distances)
        return distances
    

    def _fuzzy_distance(self, value: Union[float, int], q_low: Union[float, int], q_high: Union[float, int]) -> Union[float, int]:
        """
        Calcula la Fuzzy Distance entre un valor y el rango de valores dados.

        Args:
            - value -> valor del gusto del usuario.
            - q_low -> límite inferior del rango del gusto.
            - q_high -> límite superior del rango del gusto.

        Return:
            - Valor de la Fuzzy Distance entre el valor y el rango ingresado.
        """
        # Distancia escalada contra límite inferior
        if value < q_low:
            return (q_low - value) / (q_high - q_low)
        # Distancia escalada contra límite superior
        elif value > q_high:
            return (value - q_high) / (q_high - q_low)
        # Valor dentro del rango (distancia = 0)
        else:
            return 0.0


    def simulate_user_choice(
            self, scored_df: pd.DataFrame, user_input: Dict, top_n: int=20, d: float=.075
        ) -> Union[Dict[str, Union[pd.Series, pd.DataFrame]], None]:
        """
        Elige los top_n vinos con mayor score.

        Parameters:
            - scored_df -> DataFrame con vinos y sus scores globales calculados.
            - user_input -> Diccionario con selección del usuario.

        Returns:
            - Diccionario con:
                - selected_wine -> Serie del vino elegido.
                - user_choices -> DataFrame con la elección posible del usuario.
                - user_wine_base -> DataFrame con todos los vinos filtrados por el usuario.
                - global_wine_base -> DataFrame con todos los vinos disponibles.
        """
        # User Inputs
        user_pairings = user_input.get("pairing_list")
        precio_min = user_input.get("precio_min")
        precio_max = user_input.get("precio_max")
        grape_list = user_input.get("grape_list")
        user_tastes = user_input.get("tastes")

        # Renombramiento de variables del scored_df como Globales (consideran toda la data)
        wine_base = self._rename_repeated_cols(df=scored_df, sufix="gbl") # Base utilizada para cálculo local ("por user")
        wine_base_gbl = wine_base.copy() # Copia global

        # Filtro por pairings
        if user_pairings is not None:
            wine_base = wine_base[wine_base[user_pairings].sum(axis=1)>0]

        # Filtro por precio
        if precio_min is not None:
            wine_base = wine_base[wine_base["price"]>=precio_min]
        if precio_max is not None:
            wine_base = wine_base[wine_base["price"]<=precio_max]
        
        # Filtro por uvas
        if grape_list is not None:
            wine_base = wine_base[wine_base[grape_list].sum(axis=1)>0]

        # Solo se ejecuta si existen vinos para la selección del usuario
        if len(wine_base) > 0:
            # Quality/price local
            wine_base = self._df_add_qual_price(wine_base)

            # TODO: Quitar redundancia de cálculo de similaridad y wine_score
            # User similarity (Fuzzy Distance) local
            wine_base = self._df_add_user_similarity(wine_base, user_tastes)

            # Puntaje sintético local del vino
            wine_base = self._score_wines(wine_base, user_input) # Puntaje "crudo"

            if len(wine_base) > max(top_n//2, 10):
                # Si el wine_base es lo suficientemente grande, el puntaje es escalado considerando la selección de filtros del usuario
                wine_base["wine_score_scld"] = self.mm_scaler.fit_transform(wine_base[["wine_score"]])
            else:
                # Si el wine_base es pequeño, se considera el wine_score global (sin filtros de usuario)
                if wine_base.index.isin(wine_base_gbl.index).all():
                    wine_base["wine_score_scld"] = wine_base_gbl.loc[wine_base.index]["wine_score_scld_gbl"]
                else:
                    raise IndexError("Index in wine_base and global dataframe passed doesn't match!")


            # Selección de uno de los mejores top_n vinos
            top_wine_base = wine_base.nlargest(top_n, "wine_score")
            top_selection_weights = [(1 - d)**n for n in range(len(top_wine_base))]    # Pesos decrecientes de a d%
            top_selection_weights = [w*random.random() for w in top_selection_weights] # Agregamos factor aleatorio con suma positiva
            total_weight = sum(top_selection_weights)
            top_selection_weights = [w / total_weight for w in top_selection_weights]  # Normalizamos para interpretabilidad
            selected_wine_id = int(random.choices(top_wine_base.index, top_selection_weights, k=1)[0]) # Selección de 1 vino según vector de pesos
            selected_wine = wine_base.loc[selected_wine_id] # Localización de vino por id en la base de vinos
            return {
                "selected_wine": selected_wine,
                "user_choices": top_wine_base,
                "user_wine_base": wine_base,
                "global_wine_base": wine_base_gbl
            }
        # Se ejecuta si no existen vinos para la selección del usuario
        else:
            logging.info("No wines match user input!")
            return None


    def _rename_repeated_cols(self, df: pd.DataFrame, cols_to_rename: List[str]=None, sufix: str="gbl"):
        """
        Renombra las columnas repetidas entre el DataFrame con todos los vinos y tras aplicarle filtros del usuario.

        Args:
            - df -> DataFrame a renombrar.
            - cols_to_rename -> columnas del DataFrame a las que agregar el sufijo.
            - sufix -> sufijo a agregarle a las columnas del DataFrame.

        Returns:
            - DataFrame con columnas renombradas.
        """
        if cols_to_rename is None:
            cols_to_rename = [
                'rating_rscld',
                'price_rscld',
                'quality_price',
                'quality_price_rscld',
                'user_similarity',
                'taste_distance_raw',
                'wine_score',
                'wine_score_scld'
            ]
        renamed_cols = [f"{col_name}_{sufix}" for col_name in cols_to_rename]
        renaming_dict = dict(zip(cols_to_rename, renamed_cols))
        renamed_df = df.rename(columns=renaming_dict).copy()
        return renamed_df

    
    def simulate_like(
            self,
            global_df_scored: pd.DataFrame,
            selected_wine: pd.Series,
            global_score_name: str="wine_score_scld_gbl",
            local_score_name: str="wine_score_scld",
            alpha: float=.5,
            beta: float=.5,
            desv: float=.05
        ) -> int:
        """
        Simula si al usuario le gustó o no el vino recomendado.

        Args:
            - global_df_scored -> DataFrame que contiene el global score escalado.
            - selected_wine -> series con el wine seleccionado.
            - global_score_name -> nombre de la columna con el wine score global.
            - local_score_name -> nombre de la columna con el wine score local.
            - alpha -> multiplica la diferencia entre el score global y el local cuando es positiva.
            - beta -> multiplica la diferencia entre el score global y el local cuando es negativa.
            - desv -> desvío estandard del factor aleatorio con distribución normal y media = 0, que se le suma a la probabilidad de que le guste el vino al usuario.

        Returns:
            - liked -> 1 o 0 de acuerdo a si al usuario le gustó o no el vino.
            - prob_like -> probabilidad estimada de que al usuario el guste el vino.
        """
        global_score_norm = global_df_scored.loc[selected_wine.name][global_score_name]
        local_score_norm = selected_wine[local_score_name]
        desvio_score = global_score_norm - local_score_norm
        ajuste = 0

        if desvio_score > 0: # El vino es mejor globalmente de lo que aparenta ser localmente
            ajuste = desvio_score * alpha
        elif desvio_score < 0: # El vino es peor globalmente de lo que aparenta ser localmente
            ajuste = desvio_score * beta
        else: # El vino es igual de bueno global y localmente
            ajuste = 0

        prob_like = local_score_norm + ajuste + np.random.normal(0, desv) # Score ajustado con factor random
        prob_like = min(max(prob_like, 0),.99)
        random_prob = random.random()
        liked = 0

        if random_prob < prob_like:
            liked = 1

        return liked, prob_like


    def simulate_user_behaviour(
            self,
            user_id: int,
            first_start_date: datetime=datetime(2022,1,1),
            first_end_date: datetime=datetime(2024,1,1),
            last_timestamp: datetime=None,
            avg_days_between_purchases: int=30,
            std_days_between_purchases: int=5
        ) -> Dict[str, Union[int, None, Dict, float]]:
        """
        Simula el comportamiento completo de un usuario individual.
            - Momento de la compra.
            - Evaluación de vinos.
            - Elección de vino.
            - Opinión del vino (like = 1 | 0).

        Args:
            - user_id -> id del usuario a simular.
            - first_start_date -> fecha mínima para seteo de fecha de primera interacción.
            - first_end_date -> fecha máxima para seteo de fecha de primera interacción.
            - last_timestamp -> fecha de interacción anterior del usuario.
            - avg_days_between_purchases -> días promedio entre compra y compra de cada usuario.
                - Selección final según distribución normal truncada -> (mínimo  = 1 & máximo = datetime.now()).
            - std_days_between_purchases -> desvío estandar de días de avg_days_between_purchases.

        Returns:
            - Diccionario con:
                - event_timestamp -> momento de la interacción.
                - user_id -> id del usuario.
                - wine_id -> id del vino elegido.
                - user_input -> diccionario con preferencias del usuario.
                - metrics_local -> diccionario con métricas de evaluación según filtros del usuario.
                - metrics_global -> diccionario con métricas de evaluación de todos los vinos.
                - liked -> si al usuario le gustó o no el vino (1 | 0).
                - prob_like -> probabilidad de que al usuario le guste el vino.
        """
        user_data = None
        
        # Simulación de comportamiento de usuario
        user_input = self.generate_user_input()
        tra_df = self.tra_df
        tra_df = self._df_add_qual_price(tra_df)
        tra_df = self._score_wines(tra_df, user_input)
        user_choice_pack = self.simulate_user_choice(tra_df, user_input)

        # Asignación de timestamp
        # https://docs.python.org/3/library/datetime.html#timedelta-objects
        if last_timestamp is None: 
        # Fecha de primera interacción
            days_range = (first_end_date - first_start_date).days
            timestamp = first_start_date + timedelta(days=np.random.randint(0, days_range))
        else:
        # Fecha de siguiente interacción
            days_until_next_purchase = int(
                np.clip(np.random.normal(avg_days_between_purchases, std_days_between_purchases), 1, datetime.now())
            )
            timestamp = last_timestamp + timedelta(days=days_until_next_purchase)

        if user_choice_pack is None:
            user_data = {
                "event_timestamp": timestamp,
                "user_id": user_id,
                "wine_id": None,
                "user_input": user_input,
                "metrics_local": None,
                "metrics_global": None,
                "liked": None,
                "prob_like": None
            }
        else:
            selected_wine = user_choice_pack["selected_wine"]
            tra_df = user_choice_pack["global_wine_base"]
            liked, prob_like = self.simulate_like(tra_df, selected_wine)
            user_data = {
                "event_timestamp": timestamp,
                "user_id": user_id,
                "wine_id": int(selected_wine.name),
                "user_input": user_input,
                "metrics_local": selected_wine["rating_rscld":],
                "metrics_global": selected_wine["rating_rscld_gbl":"wine_score_scld_gbl"],
                "liked": liked,
                "prob_like": prob_like
            }
        
        return user_data
    

    def simulate_user_repurchases(
            self,
            user_first_pick_data: Dict,
            avg_days_between_purchases: int=30,
            std_days_between_purchases: int=5,
            liked_churn_rate: float=.05,
            disliked_churn_rate: float=.4,
            max_repurchases: int=20
        ) -> int:
        """
        Simula las recompras de un usuario mientras está activo en la app.
            - Tomar assumption de cada cuanto un usuario realiza sus compras en promedio.

        Args:
            - user_first_pick_data -> DataFrame con la primera selección del usuario.
            - avg_days_between_purchases -> días promedio entre compra y compra de cada usuario.
                - Selección final según distribución normal truncada -> (mínimo  = 1 & máximo = datetime.now()).
            - std_days_between_purchases -> desvío estandar de días de avg_days_between_purchases.
            - liked_churn_rate -> La probabilidad de que el usuario churnee habiéndole gustado la última selección.
            - disliked_churn_rate -> La probabilidad de que el usuario churnee habiéndole disgustago al última selección.
            - max_repurchases -> La cantidad máxima de recompras que puede tener un  usuario.

        Returns:
            - Historial de interacciones del usuario con la app.
        """
        user_history = user_first_pick_data
        timestamp = user_first_pick_data.loc[0, "timestamp"]
        user_id = user_first_pick_data.loc[0, "user_id"]
        liked = user_first_pick_data.loc[0, "liked"]
        repurchase_count = 0
        
        while repurchase_count <= max_repurchases:
            churn_prob = liked_churn_rate if liked == 1 else disliked_churn_rate
            if np.random.rand() > churn_prob:
                # Creación de recompra
                user_data = self.simulate_user_behaviour(
                    user_id=user_id,
                    last_timestamp=timestamp,
                    avg_days_between_purchases=avg_days_between_purchases,
                    std_days_between_purchases=std_days_between_purchases
                    )
                # Adición a la historia del user
                user_history = pd.concat(user_data, ignore_index=True)
                # Actualizamos parámetros para próxima recompra
                timestamp = user_data.loc[0, "timestamp"]
                liked = 0 if user_data.loc[0, "liked"] is None else user_data.loc[0, "liked"]
                repurchase_count += 1
            else:
                break

        
        return user_history


    # TODO: explorar @dataclass para no tener que pasar los mismos argumentos una y otra vez.
    def generate_synthetic_data(
            self,
            n_users: int,
            first_start_date: datetime=datetime(2022,1,1),
            first_end_date: datetime=datetime(2024,1,1),
            repurchase: bool = False,
            avg_days_between_purchases: int=30,
            std_days_between_purchases: int=5,
            liked_churn_rate: float=0.05,
            disliked_churn_rate: float=0.40,
            max_repurchases: int=20,
            verbose: bool = True
        ) -> pd.DataFrame:
        """
        Genera datos sintéticos para n usuarios.

        Parameters:
            - n_users -> Cantidad de usuarios sintéticos a simular.
            - repurchase -> Posibilidad de que los usuarios vuelvan a comprar.
            - verbose -> Mostrar progreso de simulación.

        Returns:
            - DataFrame con columnas
                - user_id -> id del usuario que realiza la selección.
                - wine_id -> id del vino seleccionado
                - user_input -> Diccionario con input del usuario, , liked
        """
        synthetic_data = []

        if verbose:
            print(f"Iniciando simulación de {n_users} usuarios {"con" if repurchase else "sin"} recompra...")

        for i in range(n_users):
            if repurchase:
                # First user interaction with the app
                user_data = self.simulate_user_behaviour(
                    user_id=i,
                    first_start_date=first_start_date,
                    first_end_date=first_end_date
                )
                # Adds all repurchases to user first interaction
                user_history = self.simulate_user_repurchases(
                    user_first_pick_data=user_data,
                    avg_days_between_purchases=avg_days_between_purchases,
                    std_days_between_purchases=std_days_between_purchases,
                    liked_churn_rate=liked_churn_rate,
                    disliked_churn_rate=disliked_churn_rate,
                    max_repurchases=max_repurchases
                )
                # Appending all user interaction history
                synthetic_data.append(user_history)
            else:
                user_data = self.simulate_user_behaviour(user_id=i)
                synthetic_data.append(user_data)
            
            if verbose:
                # Barra de progreso que se actualiza en la misma línea
                completed = i + 1
                percentage = (completed / n_users) * 100
                bar_length = 30
                filled_length = int(bar_length * completed / n_users)

                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                # \r vuelve al inicio de la línea, end='' evita nueva línea
                print(f'\rSimulaciones Completadas: {completed}/{n_users} ({percentage:.1f}%) [{bar}]', 
                    end='', flush=True)
                
        if verbose: 
            print("\n✅ Simulación Terminada con Éxito!")

        return pd.DataFrame(synthetic_data)

