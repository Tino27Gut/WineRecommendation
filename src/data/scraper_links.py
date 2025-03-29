# Standard Libraries
import logging
import os
import sys
import time

# Third Party Libraries
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
#from selenium import webdriver
#from selenium.webdriver.chrome.service import Service
#from webdriver_manager.chrome import ChromeDriverManager

# Own Libraries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from utils import setup_driver


def apply_filters(driver, wait, sleep_secs=2):
    """Aplica los filtros en la página de Vivino."""
    try:
        wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "simpleLabel-module__selectedKey--3ngzL"))).click()
        wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@data-value='us']"))).click()
        time.sleep(sleep_secs) # Esperar a que se actualice la página
        
        red_wine_label = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[@data-testid='wineTypes_1']")))
        driver.execute_script("arguments[0].click();", red_wine_label)
        
        slider_1 = driver.find_element(By.XPATH, "//div[@class='rc-slider-handle rc-slider-handle-1']")
        slider_2 = driver.find_element(By.XPATH, "//div[@class='rc-slider-handle rc-slider-handle-2']")
        
        actions = ActionChains(driver)
        actions.click_and_hold(slider_1).move_by_offset(-10, 0).release().perform()
        actions.click_and_hold(slider_2).move_by_offset(150, 0).release().perform()
        
        any_rating_label = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[.//input[@name='rating' and @value='1']]")))
        driver.execute_script("arguments[0].click();", any_rating_label)
        
        countries_dropdown = driver.find_element(By.XPATH, "//legend[@data-testid='filter-toggle-button'][.//h5[.='Countries']]")
        driver.execute_script("arguments[0].click();", countries_dropdown)
        
        countries_ar = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[@data-testid='countries_ar']")))
        driver.execute_script("arguments[0].click();", countries_ar)
        
        logging.info("Filtros aplicados correctamente.")
    except Exception as e:
        logging.error(f"Error aplicando filtros: {e}")


def get_page_links(driver, wait):
    """Extrae los links de los vinos en la página actual."""
    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        links = ["https://www.vivino.com" + a["href"] for a in soup.find_all('a', {"data-testid": "vintagePageLink"})]
        
        next_page_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@data-trackingid='buttonDefault'][.//span[.='Next']]") ))
        if next_page_btn.get_attribute("aria-disabled") == "true":
            return links, False
        
        driver.execute_script("arguments[0].click();", next_page_btn)
        return links, True
    except Exception as e:
        logging.error(f"Error obteniendo links: {e}")
        return [], False


def scrape_links(output_path="../src/data/raw/links/wine_links_test.csv", headless=False, sleep_secs=5):
    """Función principal para obtener todos los links de vinos."""
    driver = setup_driver(headless)
    wait = WebDriverWait(driver, 10)
    driver.get("https://www.vivino.com/explore")
    
    apply_filters(driver, wait)
    time.sleep(sleep_secs)  # Esperar a que la página actualice resultados
    
    link_list = []
    next_page_available = True
    while next_page_available:
        links, next_page_available = get_page_links(driver, wait)
        link_list.extend(links)
    
    pd.DataFrame(link_list, columns=["wine_link"]).to_csv(output_path, index=False)
    logging.info(f"Scraping completado. Links guardados en {output_path}")
    driver.quit()
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scrape_links()
