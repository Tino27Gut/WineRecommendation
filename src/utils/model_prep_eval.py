from typing import Union, List, Dict, Tuple, Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC


def oversample_df(
        df: pd.DataFrame,
        sampling_object: object,
        random_state: Union[int, None] = None,
        categorical_features: Union[List[str], None] = None,
        class_col: str = "liked"    
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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