from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

url = "http://www.ufcstats.com/fighter-details/93fe7332d16c6ad9"

driver.get(url)
time.sleep(3)

soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# Find career stats
print("=== CAREER STATS SECTION ===")
career_stats = soup.find("div", class_="b-list__info-box")
if career_stats:
    print(career_stats.prettify()[:2000])
else:
    print("Not found - trying alternative...")
    # Print all div classes on the page so we can find the right one
    divs = soup.find_all("div")
    classes = set()
    for div in divs:
        if div.get("class"):
            classes.add(str(div.get("class")))
    for c in sorted(classes):
        print(c)

print("\n=== FIGHT HISTORY SECTION ===")
fight_table = soup.find("table", class_="b-fight-details__table")
if fight_table:
    # Just print the first fight row
    rows = fight_table.find_all("tr")
    for row in rows[:3]:
        print(row.prettify()[:1000])
        print("---")
else:
    print("Not found - trying alternative...")
    tables = soup.find_all("table")
    for t in tables:
        print("Table classes:", t.get("class"))