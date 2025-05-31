import pandas as pd
import os

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


def manage_outlier_IQR(df, i=1.5, func="find"):
    """
    Devuelve un DataFrame con o sin outliers en base el parametro 'func'.\n
    source: #https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/\n
    df = DataFrame\n
    i = Indice que multiplica IQR (default = 1.5)
    func = 'find' para obtener solo outliers, 'remove' para quitarlos (default = 'find')
    """
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3-q1
    if func == "find":
        outliers = (df < (q1 - i * iqr)) | (df > (q3 + i * iqr))
    elif func == "remove":
        outliers = (df >= (q1 - i * iqr)) & (df <= (q3 + i * iqr))
    else:
        raise AttributeError("Please select 'find' or 'remove' funcions!")
    return df[outliers]



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
