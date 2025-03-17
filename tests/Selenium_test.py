from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Set up options for headless mode
options = Options()
options.headless = True

# Explicitly specify the Chrome binary location
options.binary_location = "/opt/google/chrome/google-chrome"

# Use webdriver-manager to get the chromedriver binary
service = Service(ChromeDriverManager().install())

# Initialize the Chrome WebDriver with the Service and options
driver = webdriver.Chrome(service=service, options=options)

driver.get("https://google.com/")
print(driver.title)
driver.quit()
