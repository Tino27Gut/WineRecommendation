from typing import Union, List, Dict, Tuple, Literal, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.base import clone
from scipy import stats


# ==============================
# OVERSAMPLING
# ==============================
def oversample_df(
        df: pd.DataFrame,
        sampling_object: object,
        random_state: Union[int, None] = None,
        categorical_features: Union[List[str], None] = None,
        class_col: str = "liked"    
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Oversamplea un DataFrame utilizando un objeto de oversampling proporcionado 
    (por ejemplo, SMOTE o SMOTENC).

    Args:
        - df -> DataFrame original que contiene las features y la columna de clase.
        - sampling_object -> Clase de oversampling que se va a utilizar. Debe tener `.fit_resample()`.
        - random_state -> Semilla para reproducibilidad. Opcional.
        - categorical_features -> Lista de nombres de columnas categóricas. 
            - Solo requerido para SMOTENC.
        - class_col -> Nombre de la columna que contiene la clase objetivo (por defecto: "liked").

    Returns:
        - Tupla de DataFrames
            - X_resampled -> DataFrame con las features reamostradas.
            - y_resampled -> Serie con la clase objetivo reamostrada.
    """
    if sampling_object.__name__ == SMOTENC.__name__:
        sampler = sampling_object(
            categorical_features=categorical_features,
            random_state=random_state
        )
    else:
        sampler = sampling_object(random_state=random_state)

    X = df.drop(columns=[class_col])
    y = df[class_col]

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    return X_resampled, y_resampled


# ==============================
# MODEL PREP-TEST-DISPLAY
# ==============================
def train_test_model(
        X: Union[pd.Series, pd.DataFrame],
        y: Union[pd.Series, pd.DataFrame],
        model_object: object,
        test_size: float = 0.2,
        random_state: Union[int, None] = None,
        stratify: Union[pd.Series, pd.DataFrame, None] = None
    ) -> Dict[str, Union[str, object, float]]:
    """
    Realiza el split de los datos, entrena el modelo, evalúa en train y test, 
    y retorna métricas de performance junto con la matriz de confusión.

    Args:
        - X -> Variables independientes (features).
        - y -> Variable dependiente (target).
        - model_object -> Modelo de sklearn (u otro con .fit y .predict_proba).
        - test_size -> Proporción del set que se usará como test.
        - random_state -> Semilla para reproducibilidad.
        - stratify -> Variable para realizar stratified split (recomendado si la clase está desbalanceada).

    Returns:
        - Diccionario con los siguientes resultados
            - "train_report" -> Reporte de clasificación en train (str).
            - "train_cmax" -> Visualización de matriz de confusión (ConfusionMatrixDisplay).
            - "train_auc" -> AUC score en el set de entrenamiento (float).
            - "test_report" -> Reporte de clasificación en test (str).
            - "test_cmax" -> Visualización de matriz de confusión en test (ConfusionMatrixDisplay).
            - "test_auc" -> AUC score en el set de test (float).
            - "X_test" -> Features de test (pd.DataFrame).
            - "y_test" -> Labels de test (pd.Series).
            - "cm_train" -> Matriz de confusión de train (np.ndarray).
            - "cm_pred" -> Matriz de confusión de test (np.ndarray).
    """
    # Split de datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    
    # Entrenamiento
    model_object.fit(X_train, y_train)

    # Predicción de train
    y_train_proba = model_object.predict_proba(X_train)[:, 1]
    y_train_pred = model_object.predict(X_train)
    cm_train = confusion_matrix(y_train, y_train_pred)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=model_object.classes_)

    # Predicción de test
    y_pred_proba = model_object.predict_proba(X_test)[:, 1]
    y_pred = model_object.predict(X_test)
    cm_pred = confusion_matrix(y_test, y_pred)
    disp_pred = ConfusionMatrixDisplay(confusion_matrix=cm_pred, display_labels=model_object.classes_)

    return {
        "train_report": classification_report(y_train, y_train_pred),
        "train_cmax": disp_train,
        "train_auc": roc_auc_score(y_train, y_train_proba),
        "test_report": classification_report(y_test, y_pred),
        "test_cmax": disp_pred,
        "test_auc": roc_auc_score(y_test, y_pred_proba),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "cm_train": cm_train,
        "cm_pred": cm_pred
    }    


def train_val_test_model(
        X: Union[pd.Series, pd.DataFrame],
        y: Union[pd.Series, pd.DataFrame],
        model_object: object,
        val_size: float = 0.2,
        test_size: float = 0.2,
        random_state: Union[int, None] = None,
        stratify: Union[pd.Series, pd.DataFrame, None] = None
    ) -> Dict[str, Union[str, object, float]]:
    """
    Realiza el split de los datos, entrena el modelo, evalúa en train, validation y test, 
    y retorna métricas de performance junto con la matriz de confusión.

    Args:
        - X -> Variables independientes (features).
        - y -> Variable dependiente (target).
        - model_object -> Modelo de sklearn (u otro con .fit y .predict_proba).
        - val_size -> Proporción del set que se usará como validation.
        - test_size -> Proporción del set que se usará como test.
        - random_state -> Semilla para reproducibilidad.
        - stratify -> Variable para realizar stratified split (recomendado si la clase está desbalanceada).

    Returns:
        - Diccionario con los siguientes resultados

            - "train_report" -> Reporte de clasificación en train (str).
            - "train_cmax" -> Visualización de matriz de confusión en train (ConfusionMatrixDisplay).
            - "train_auc" -> AUC score en el set de entrenamiento (float).

            - "val_report" -> Reporte de clasificación en val (str).
            - "val_cmax" -> Visualización de matriz de confusión en val (ConfusionMatrixDisplay).
            - "val_auc" -> AUC score en el set de validación (float).

            - "test_report" -> Reporte de clasificación en test (str).
            - "test_cmax" -> Visualización de matriz de confusión en test (ConfusionMatrixDisplay).
            - "test_auc" -> AUC score en el set de test (float).

            - "X_test" -> Features de test (pd.DataFrame).
            - "y_test" -> Labels de test (pd.Series).

            - "cm_train" -> Matriz de confusión de train (np.ndarray).
            - "cm_val" -> Matriz de confución de validation (np.ndarray).
            - "cm_pred" -> Matriz de confusión de test (np.ndarray).
    """
    # Split de datos en train y test
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(
        X,
        y,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    # Entrenamiento
    model_object.fit(X_train, y_train)

    # Predicción de train
    y_train_proba = model_object.predict_proba(X_train)[:, 1]
    y_train_pred = model_object.predict(X_train)
    cm_train = confusion_matrix(y_train, y_train_pred, normalize="all")
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=model_object.classes_)

    # Predicción de val
    y_val_proba = model_object.predict_proba(X_val)[:, 1]
    y_val_pred = model_object.predict(X_val)
    cm_val = confusion_matrix(y_val, y_val_pred, normalize="all")
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=model_object.classes_)

    # Predicción de test
    y_pred_proba = model_object.predict_proba(X_test)[:, 1]
    y_pred = model_object.predict(X_test)
    cm_pred = confusion_matrix(y_test, y_pred, normalize="all")
    disp_pred = ConfusionMatrixDisplay(confusion_matrix=cm_pred, display_labels=model_object.classes_)

    return {
        # Train report, confusion matrix y AUC
        "train_report": classification_report(y_train, y_train_pred),
        "train_cmax": disp_train,
        "train_auc": roc_auc_score(y_train, y_train_proba),
        # Validation report, confusion matrix y AUC
        "val_report": classification_report(y_val, y_val_pred),
        "val_cmax": disp_val,
        "val_auc": roc_auc_score(y_val, y_val_proba),
        # Test report, confusion matrix y AUC
        "test_report": classification_report(y_test, y_pred),
        "test_cmax": disp_pred,
        "test_auc": roc_auc_score(y_test, y_pred_proba),
        # Train set
        "X_train": X_train,
        "y_train": y_train,
        # Validation set
        "X_val": X_val,
        "y_val": y_val,
        # Test set
        "X_test": X_test,
        "y_test": y_test,
        # Confusion Matrixes
        "cm_train": cm_train,
        "cm_val": cm_val,
        "cm_pred": cm_pred
    }    


def print_model_results(results: Dict[str, Union[str, object, float]]) -> None:
    """
    Imprime los resultados del modelo en consola y visualiza las matrices de confusión para 
    entrenamiento y testeo.

    Args:
        - results -> Diccionario con métricas de performance generado por la función `train_test_model`.

    Returns:
        - None
    """
    print("\n----------------------------------------------------\n")
    print("Train Metrics:\n")
    print(results["train_report"])
    results["train_cmax"].plot()
    plt.show()
    print("AUC Train:", results["train_auc"])
    print("\n----------------------------------------------------\n")
    print("Test Metrics:\n")
    print(results["test_report"])
    results["test_cmax"].plot()
    plt.show()
    print("AUC Test:", results["test_auc"])


def train_test_val_split(
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        val_size: float,
        test_size: float,
        random_state: Optional[int],
        stratify: Optional[pd.Series]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Hace un split de los datos en grupos train, validation y test

    Args:
        - X -> variables independientes.
        - y -> variable dependiente a predecir.
        - val_size -> porcentaje a usar del dataset original como datos de validación.
        - val_size -> porcentaje a usar del dataset original como datos de test.
        - random_state -> semilla para reproducibilidad.
        - stratify -> variable dependiente para mantener mismo balance de clases.

    Returns:
        - Tupla X_train, X_val, X_test, y_train, y_val, y_test
    """

    # Train test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # Calcula el valor del test_size para que val_size sea el % correcto del dataset total
    adj_val_size = val_size / (1 - test_size)

    # Setea stratify en caso que haya sido definido
    val_stratify = y_trainval if stratify is not None else None

    # Train val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=adj_val_size,
        random_state=random_state,
        stratify=val_stratify
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_learning_curves(
        X_train: Union[pd.DataFrame, pd.Series],
        X_val: Union[pd.DataFrame, pd.Series],
        y_train: Union[pd.DataFrame, pd.Series],
        y_val: Union[pd.DataFrame, pd.Series],
        base_model: object,
        params_dict: Dict[str, List],
        eco_score_dict: Dict[str, Union[float, int]]
    ) -> None:
    """
    Genera gráficos con curvas de aprendizaje para cada parámetro y valor pasado.

    Args:
        - X_train -> features para entrenar el modelo.
        - X_val -> features de entrada para predecir con el modelo.
        - y_train -> output real para entrenar el modelo.
        - y_val -> output real para comparar contra lo predicho por el modelo.
        - base_model -> modelo a tomar como base para plotear las curvas de aprendizaje.
        - params_dict -> diccionario de parámetros y valores para pasarle al base_model en cada iteración.

    Returns:
        - None -> plotea las curvas, pero no devuelve nada en si.
    """
    # Iterar sobre parámetros y valores
    for param, values in params_dict.items():

        # Crear listas de scores vacías para cada parámetro
        auc_train_scores = []
        auc_val_scores = []
        eco_train_scores = []
        eco_val_scores = []
        
        # Crear modelo con hiperparámetros default
        model = clone(base_model)
        
        for val in values:
            # Setear el valor del parámetro pasado en params_values
            setattr(model, param, val)

            # Entrenar el modelo
            model.fit(X_train, y_train)

            # Calcular y hacer append de AUC en train y validation
            pred_prob_train = model.predict_proba(X_train)[:, 1]
            auc_train = roc_auc_score(y_train, pred_prob_train)
            
            pred_prob_val = model.predict_proba(X_val)[:, 1]
            auc_val = roc_auc_score(y_val, pred_prob_val)

            auc_train_scores.append(auc_train)
            auc_val_scores.append(auc_val)

            # Obtener los parámetros para score económico
            avg_tkt = eco_score_dict["avg_tkt"]
            churn_like = eco_score_dict["churn_like"]
            churn_dislike = eco_score_dict["churn_dislike"]

            # Calcular y hacer append de economic score en train y validation
            pred_train = model.predict(X_train)
            train_cm = confusion_matrix(y_train, pred_train)
            eco_train = economic_score(train_cm, avg_tkt, churn_like, churn_dislike)

            pred_val = model.predict(X_val)
            val_cm = confusion_matrix(y_val, pred_val)
            eco_val = economic_score(val_cm, avg_tkt, churn_like, churn_dislike)

            eco_train_scores.append(eco_train)
            eco_val_scores.append(eco_val)

        # Normalización de scores (misma escala en sets de train y val)
        eco_train_scores_z = stats.zscore(eco_train_scores)
        eco_val_scores_z = stats.zscore(eco_val_scores)

        # Plotear Learning Curves
        param_scores_pairs = {
            "AUC": [auc_train_scores, auc_val_scores],
            "Model Gain Scld ($)": [eco_train_scores_z, eco_val_scores_z]
        }

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        if None in values:
            none_idx = values.index(None)
            values[none_idx] = "None"

        for i, (metric, pair) in enumerate(param_scores_pairs.items()):
            ax = axes[i]
            ax.plot(values, pair[0], label=f"Train {metric}", marker="o")
            ax.plot(values, pair[1], label=f"Validation {metric}", marker="o")
            ax.set_xlabel(param)
            ax.set_ylabel(f"{metric}")
            ax.set_title(f"Learning Curve - {param}")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()


# ==============================
# CÁLCULO DE SCORE ECONÓMICO
# ==============================
def _calc_next_exp_val(
        avg_tkt: float,
        churn_like: float,
        churn_dislike: float,
        precision: float,
        type: Literal["TN", "FN"] = "TN"
    ) -> float:
    """
    Calcula el valor esperado económico de una segunda recomendación en escenarios de TN o FN,
    considerando la precisión del modelo y las tasas de churn para cada tipo de experiencia.

    Args:
        - avg_tkt -> Valor promedio por compra del usuario.
        - churn_like -> Probabilidad de churn si al usuario le gusta el vino.
        - churn_dislike -> Probabilidad de churn si al usuario no le gusta el vino.
        - precision -> Precisión del modelo al identificar recomendaciones positivas.
        - type -> Tipo de predicción para la cual se calcula el valor ("TN" o "FN").

    Returns:
        - Valor económico esperado ajustado según la calidad de la segunda recomendación.
    """
    if type == "TN":
        actual_recommendation_exp_val = (1 - churn_dislike)
    elif type == "FN":
        actual_recommendation_exp_val = (1 - churn_like)
    else:
        raise Exception("Type is not recognized. Only TN and FN types are accepted!")
    
    next_recommendation_exp_val = avg_tkt * ((((1 - churn_like) * precision) - (churn_dislike * (1 - precision))) - actual_recommendation_exp_val)
    return next_recommendation_exp_val


def economic_score(
        conf_matrix: np.ndarray,
        avg_tkt: float,
        churn_like: float,
        churn_dislike: float
    ) -> float:
    """
    Calcula el valor económico total esperado del modelo a partir de su matriz de confusión
    y los valores de negocio definidos (churns y ticket promedio).

    Args:
        - conf_matrix -> Matriz de confusión con formato [TN, FP, FN, TP].
        - avg_tkt -> Valor promedio por compra del usuario.
        - churn_like -> Probabilidad de churn si al usuario le gusta el vino.
        - churn_dislike -> Probabilidad de churn si al usuario no le gusta el vino.

    Returns:
        - Valor económico total esperado del modelo.
    """
    tn, fp, fn, tp = conf_matrix.ravel()

    true_precision = tp / (tp + fp)

    val_tn = _calc_next_exp_val(avg_tkt, churn_like, churn_dislike, true_precision, type="TN")
    cost_fp = - avg_tkt * churn_dislike
    cost_fn = _calc_next_exp_val(avg_tkt, churn_like, churn_dislike, true_precision, type="FN")
    val_tp = avg_tkt * (1 - churn_like)

    score = (
        tn * val_tn +
        fp * cost_fp + 
        fn * cost_fn +
        tp * val_tp
    )

    return score