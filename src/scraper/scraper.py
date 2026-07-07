from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

# ── helpers ──────────────────────────────────────────────────────────────────

def make_driver():
    """Launch a headless Chrome browser (no visible window)."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")          # runs in background
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver


def get_soup(driver, url, wait=2):
    """Load a URL and return a BeautifulSoup object."""
    driver.get(url)
    time.sleep(wait)
    return BeautifulSoup(driver.page_source, "html.parser")


# ── step 1: collect all fighter URLs ─────────────────────────────────────────

def get_fighter_urls(driver):
    """Scrape the A-Z index and return a list of all fighter URLs."""
    base = "http://www.ufcstats.com/statistics/fighters?char={}&page=all"
    letters = "abcdefghijklmnopqrstuvwxyz"
    fighter_urls = []

    for letter in letters:
        print(f"Collecting fighters for letter: {letter.upper()}")
        soup = get_soup(driver, base.format(letter), wait=2)

        rows = soup.find_all("tr", class_="b-statistics__table-row")
        for row in rows:
            link = row.find("a", class_="b-link b-link_style_black")
            if link and link.get("href"):
                fighter_urls.append(link["href"])

        time.sleep(1)  # be polite to the server

    # remove duplicates
    fighter_urls = list(set(fighter_urls))
    print(f"\nTotal unique fighter URLs collected: {len(fighter_urls)}")
    return fighter_urls


# ── step 2: scrape individual fighter page ────────────────────────────────────

def scrape_fighter(driver, url):
    """Scrape a single fighter page and return a dict of their data."""
    soup = get_soup(driver, url, wait=2)

    fighter = {"url": url}

    # --- name ---
    name_tag = soup.find("span", class_="b-content__title-highlight")
    fighter["name"] = name_tag.text.strip() if name_tag else None

    # --- record ---
    record_tag = soup.find("span", class_="b-content__title-record")
    fighter["record"] = record_tag.text.strip().replace("Record: ", "") if record_tag else None

    # --- physical stats and DOB ---
    info_items = soup.find_all("li", class_="b-list__box-list-item b-list__box-list-item_type_block")
    for item in info_items:
        label_tag = item.find("i", class_="b-list__box-item-title")
        if not label_tag:
            continue
        label = label_tag.text.strip().rstrip(":").lower()
        value = item.text.replace(label_tag.text, "").strip()
        if label == "height":
            fighter["height"] = value
        elif label == "weight":
            fighter["weight"] = value
        elif label == "reach":
            fighter["reach"] = value
        elif label == "stance":
            fighter["stance"] = value
        elif label == "dob":
            fighter["dob"] = value

    # --- career statistics ---
    stat_items = soup.find_all("li", class_="b-list__box-list-item")
    for item in stat_items:
        label_tag = item.find("i")
        if not label_tag:
            continue
        label = label_tag.text.strip().rstrip(":").lower().replace(" ", "_").replace(".", "")
        value = item.text.replace(label_tag.text, "").strip()
        if label in ["slpm", "str_acc", "sapm", "str_def",
                     "td_avg", "td_acc", "td_def", "sub_avg"]:
            fighter[label] = value

    # --- fight history URLs ---
    fight_rows = soup.find_all(
        "tr",
        class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click"
    )
    fight_urls = []
    for row in fight_rows:
        link = row.get("data-link")
        if link:
            fight_urls.append(link)

    fighter["fight_urls"] = fight_urls
    fighter["total_fights"] = len(fight_urls)

    return fighter


# ── step 3: test on a small sample ───────────────────────────────────────────

def main():
    driver = make_driver()

    print("=== STEP 1: Collecting fighter URLs for letter A only (test run) ===")
    base = "http://www.ufcstats.com/statistics/fighters?char=a&page=all"
    soup = get_soup(driver, base, wait=3)

    rows = soup.find_all("tr", class_="b-statistics__table-row")
    test_urls = []
    for row in rows:
        link = row.find("a", class_="b-link b-link_style_black")
        if link and link.get("href"):
            test_urls.append(link["href"])

    print(f"Found {len(test_urls)} fighters for letter A")

    print("\n=== STEP 2: Scraping first 5 fighters as a test ===")
    fighters = []
    for i, url in enumerate(test_urls[:5]):
        print(f"Scraping fighter {i+1}/5: {url}")
        data = scrape_fighter(driver, url)
        fighters.append(data)
        print(f"  Name: {data.get('name')} | Record: {data.get('record')} | Fights found: {data.get('total_fights')}")
        time.sleep(1)

    driver.quit()

    print("\n=== STEP 3: Saving to CSV ===")
    os.makedirs("data/raw", exist_ok=True)

    # save without fight_urls column for readability
    df = pd.DataFrame(fighters)
    df_display = df.drop(columns=["fight_urls"])
    df_display.to_csv("data/raw/fighters_test.csv", index=False)
    print("Saved to data/raw/fighters_test.csv")
    print("\nSample data:")
    print(df_display.to_string())


if __name__ == "__main__":
    main()