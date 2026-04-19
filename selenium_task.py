"""
Selenium: login on a public demo site, then extract text (meets "login + data extraction").

Demo credentials are published by the site for testing only:
https://the-internet.herokuapp.com/login

Requires: pip install selenium

Browser: Microsoft Edge only (Selenium 4+ resolves msedgedriver). Not Chrome or Brave.
"""
import time

from selenium import webdriver
from selenium.webdriver.common.by import By


LOGIN_URL = "https://the-internet.herokuapp.com/login"
USER = "tomsmith"
PASSWORD = "SuperSecretPassword!"


def main() -> None:
    options = webdriver.EdgeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Edge(options=options)
    try:
        driver.get(LOGIN_URL)
        time.sleep(1)

        driver.find_element(By.ID, "username").send_keys(USER)
        driver.find_element(By.ID, "password").send_keys(PASSWORD)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        time.sleep(1)

        flash = driver.find_element(By.ID, "flash").text
        print("After login, banner text:")
        print(flash.strip())

        driver.get("https://the-internet.herokuapp.com/")
        time.sleep(1)
        heading = driver.find_element(By.CSS_SELECTOR, "h1.heading").text
        print("\nHome heading extracted:")
        print(heading)
    finally:
        driver.quit()
        print("\nBrowser closed.")


if __name__ == "__main__":
    main()
