import pandas as pd
import os
from typing import List, Dict, Tuple, Union

# Scrapper
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Machine Learning
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

def save_csv(df, path=None, filename="output.csv"):
    """ Guarda un DataFrame como CSV en la ruta especificada. """
    if path is None:
        path = os.getcwd()
    full_path = os.path.join(path, filename)
    df.to_csv(full_path, index=False)
    print(f"Archivo guardado en: {full_path}")
    

def setup_driver(headless=True):
    """ Configura el driver de selenium para WebScraping. """
    options = webdriver.ChromeOptions()
    options.headless = headless
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(300)
    return driver


def manage_outlier_IQR(data: Union[pd.DataFrame, pd.Series], i: float=1.5, func: str="find") -> Union[pd.DataFrame, pd.Series]:
    """
    Devuelve un DataFrame con o sin outliers en base el parametro 'func'.\n
    source: #https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/\n

    Args:
        - df = DataFrame\n
        - i = Indice que multiplica IQR (default = 1.5)
        - func = 'find' para obtener solo outliers, 'remove' para quitarlos (default = 'find')

    Returns:
        - DataFrame o Serie con o sin outliers (según func).
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3-q1
    if func == "find":
        outliers = (data < (q1 - i * iqr)) | (data > (q3 + i * iqr))
    elif func == "remove":
        outliers = (data >= (q1 - i * iqr)) & (data <= (q3 + i * iqr))
    else:
        raise AttributeError("Please select 'find' or 'remove' funcions!")
    return data[outliers]



def eval_cluster(X, labels):
    """
    Calcula métricas de performance de clusters.

    Atributos:
    - X: Matriz de variables utilizadas para clusterizar.
    - labels: Array de clusters calculado a partir de X.

    Return:
    - Devuelve un diccionario con las métricas "Silhouette", "DB Score" y "CH Score".
    """

    return {
        "Silhouette": silhouette_score(X, labels),
        "DB Score": davies_bouldin_score(X, labels),
        "CH Score": calinski_harabasz_score(X, labels),
    }
