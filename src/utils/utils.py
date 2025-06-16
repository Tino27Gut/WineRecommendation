import os
from typing import List, Dict, Tuple, Union
from pathlib import Path
import pandas as pd

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


def get_project_root() -> Path:
    """
    Encuentra la raíz del proyecto usando múltiples estrategias.

    Returns:
        - Devuelve el directorio raíz del proyecto
    """
    current = Path(__file__).resolve().parent
    
    # Estrategia 1: Buscar marcadores de proyecto
    root_markers = ["pyproject.toml", ".gitignore"]
    
    # Estrategia 2: Buscar estructura específica del proyecto
    def has_project_structure(path):
        return (
            (path / "src").is_dir() and
            (path / "src" / "data").is_dir() and
            (path / "src" / "models").is_dir() and
            (path / "src" / "utils").is_dir()
        )
    
    # Buscar hacia arriba
    while current != current.parent:
        
        # Estrategia 1: Archivos marcadores
        for marker in root_markers:
            if (current / marker).exists():
                return current
        
        # Estrategia 2: Estructura del proyecto
        if has_project_structure(current):
            return current
            
        current = current.parent
    
    raise Exception(
        "No se encontró la raíz del proyecto. Opciones:\n"
        "1. Crear archivos pyproject.toml y .gitignore en la root del proyecto (directorio principal)\n"
        "2. Asegurar estructura src/data/, src/models/ y src/utils/"
    )

def get_project_file_path(*path_parts):
    """
    Construye rutas relativas al proyecto.
    
    Args:
        - *path_parts -> Partes de la ruta como argumentos separados
        
    Returns:
        - Path -> Ruta completa al archivo/directorio
        
    Ejemplo:
        get_project_path("src", "data", "transformed", "wines.csv")
    """
    return get_project_root().joinpath(*path_parts)

