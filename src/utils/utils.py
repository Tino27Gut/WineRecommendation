import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

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

