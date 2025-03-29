# Standard Libraries
import logging
import os
import sys
import time

# Third Party Libraries
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# Own Libraries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
import utils as ut


def scrape_wine_data(driver, link_file_name="wine_links.csv", import_path="src/data/raw/links/", scraped_file_name="wines.csv", save_path="src/data/raw/scraped_wines/"):
    wine_df = pd.read_csv(import_path + link_file_name).drop_duplicates()
    wine_df["name"] = wine_df["year"] = wine_df["winery"] = wine_df["rating"] = None
    wine_df["rating_qty"] = wine_df["price"] = wine_df["body"] = wine_df["tannis"] = None
    wine_df["sweetness"] = wine_df["acidity"] = wine_df["notes"] = wine_df["pairings"] = None
    wine_df["grapes"] = wine_df["region"] = wine_df["style"] = wine_df["image"] = None

    first_link = True

    for index, row in wine_df.iterrows():
        driver.get(row["wine_link"])

        if first_link:
            # Step 1: click country dropdown
            time.sleep(2)
            wait = WebDriverWait(driver, 30)
            dropdown = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "simpleLabel-module__selectedKey--3ngzL")))
            dropdown.click()
            time.sleep(2)
            # Step 2: select "United States" from the list
            country_option = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@data-value='us']")))
            country_option.click()
            first_link = False
        
        time.sleep(6)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        try: # DATA 1: NOMBRE (name)
            wine_name_tag = soup.find("a", {"data-cartitemsource": "wine-page-master-link"})
            wine_name = wine_name_tag.get_text(strip=True)
            wine_df.at[index, "name"] = wine_name
        except: pass

        try: # DATA 2: AÑO (year)
            wine_df.at[index, "year"] = wine_name_tag.find_parent("div").get_text(strip=True).replace(wine_name, "").strip()
        except: pass

        try: # DATA 3: BODEGA (WINERY)
            wine_df.at[index, "winery"] = soup.find("h1").find("div").find("a").find("div").get_text(strip=True)
        except: pass

        try: # DATA 4: REVIEWS (Rating)
            rating_tag = soup.find("a", {"href": "#all_reviews"})
            wine_df.at[index, "rating"] = rating_tag.find("div").find("div").get_text(strip=True)
            wine_df.at[index, "rating_qty"] = rating_tag.find_all("div")[-1].get_text(strip=True).split(" ")[0]
        except: pass

        try: # DATA 5: PRICE (Precio)
            wine_df.at[index, "price"] = soup.find("span", {"class": "purchaseAvailability__currentPrice--3mO4u"}).get_text(strip=True).replace("$", "")
        except: pass

        try: # DATA 6: TASTE (Sabor)
            wine_taste = soup.find_all("tr", {"class": "tasteStructure__tasteCharacteristic--jLtsE"})
            tastes_values = {"body": None, "tannis": None, "sweetness": None, "acidity": None}

            for taste in wine_taste:
                name = taste.find("td").get_text(strip=True)
                bar = taste.find("span", {"class", "indicatorBar__progress--3aXLX"})["style"]
                val = bar.split(";")[1].split(":")[1]
                
                if name == "Light":
                    tastes_values["body"] = val
                elif name == "Smooth":
                    tastes_values["tannis"] = val
                elif name == "Dry":
                    tastes_values["sweetness"] = val
                elif name == "Soft":
                    tastes_values["acidity"] = val

            wine_df.at[index, "body"] = tastes_values["body"]
            wine_df.at[index, "tannis"] = tastes_values["tannis"]
            wine_df.at[index, "sweetness"] = tastes_values["sweetness"]
            wine_df.at[index, "acidity"] = tastes_values["acidity"]
        except Exception as e:
            logging.error(f"Error buscando sabores: {e}")


        # DATA 7 : NOTE MENTIONS (Menciones de notas)

        try:
            mentions = soup.find_all("div", {"data-testid": "mentions"})

            mentions_dict = {}

            for mention in mentions:
                note = mention.find("span").get_text(strip=True)
                n_mentions = mention.get_text().split(" ")[0]
                mentions_dict[str(note)] = int(n_mentions)
        except:
            mentions_dict = None
        
        # DATA 8 : PAIRINGS (Maridajes)
        try:
            pairing_tags = soup.find("div", {"class": "foodPairing__foodContainer--1bvxM"}).find_all("a", recursive=False)
            
            pairing_list = []

            for i in pairing_tags:
                pairing = i.find("div", {"role": "img"})["aria-label"].lower()
                pairing_list.append(pairing)
        except:
            pairing_tags = None


        # DATA 9 : GRAPES (Uvas)
        try:
            wine_facts = soup.find_all("tr", {"data-testid": "wineFactRow"})
        except:
            wine_facts = None

        try:
            grapes = list(wine_facts[1].find("td").get_text(strip=True).split(","))
        except:
            grapes = None

        try:
        # DATA 10 : REGION
            region = list(wine_facts[2].find("td").get_text(strip=True).split("/"))
        except:
            region = None

        try:
            # DATA 11 : STYLE (Estilo)
            style = wine_facts[3].find("td").get_text()
        except:
            style = None

        try:
            # DATA 12 : WINE IMAGE (Imagen del Vino)
            img = "https:" + soup.find("img", {"class": "wineLabel-module__image--3HOnd"})["src"]
        except:
            img = None
        
        # ÚLTIMO PASO: GUARDAR EN DATAFRAME
        wine_df.at[index, "notes"] = mentions_dict
        wine_df.at[index, "pairings"] = pairing_list
        wine_df.at[index, "grapes"] = grapes
        wine_df.at[index, "region"] = region
        wine_df.at[index, "style"] = style
        wine_df.at[index, "image"] = img

    driver.quit()

    ut.save_csv(wine_df, save_path, scraped_file_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    driver = ut.setup_driver()
    scrape_wine_data(driver, link_file_name="wine_links_copy.csv")

