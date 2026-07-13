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


# ── step 3:  run it ───────────────────────────────────────────

def main():
    driver = make_driver()

    os.makedirs("data/raw", exist_ok=True)

    # ── load fighter URLs ─────────────────────────────────────────────────────
    print("=== STEP 1: Loading fighter URLs ===")
    with open("data/raw/fighter_urls.txt", "r") as f:
        all_urls = [line.strip() for line in f.readlines()]
    print(f"Total fighters to scrape: {len(all_urls)}")

    # ── figure out where we left off ─────────────────────────────────────────
    already_scraped = set()
    existing_fighters = []

    # sort checkpoint files NUMERICALLY not alphabetically
    checkpoint_files = [
        f for f in os.listdir("data/raw")
        if f.startswith("fighters_checkpoint_") and f.endswith(".csv")
    ]

    if checkpoint_files:
        # sort by the number in the filename
        checkpoint_files.sort(key=lambda x: int(x.replace("fighters_checkpoint_", "").replace(".csv", "")))
        latest = checkpoint_files[-1]
        latest_num = int(latest.replace("fighters_checkpoint_", "").replace(".csv", ""))
        print(f"Found latest checkpoint: {latest} ({latest_num} fighters)")

        df_existing = pd.read_csv(f"data/raw/{latest}")
        existing_fighters = df_existing.to_dict("records")
        already_scraped = set(df_existing["url"].tolist())
        print(f"Already scraped: {len(already_scraped)} fighters")
    else:
        print("No checkpoint found — starting from beginning")

    # filter to only URLs we have not scraped yet
    remaining_urls = [u for u in all_urls if u not in already_scraped]
    print(f"Remaining to scrape: {len(remaining_urls)} fighters")

    if len(remaining_urls) == 0:
        print("All fighters already scraped — saving final file")
        driver.quit()
        df = pd.DataFrame(existing_fighters)
        df.to_csv("data/raw/all_fighters.csv", index=False)
        print(f"Saved {len(existing_fighters)} fighters to data/raw/all_fighters.csv")
        return

    # ── scrape remaining fighters ─────────────────────────────────────────────
    print("\n=== STEP 2: Scraping remaining fighters ===")
    fighters = existing_fighters.copy()
    failed = []

    for i, url in enumerate(remaining_urls):
        try:
            print(f"[{i+1}/{len(remaining_urls)}] Scraping: {url}")
            data = scrape_fighter(driver, url)
            fighters.append(data)

            # save checkpoint every 100 fighters
            if (i + 1) % 100 == 0:
                df_temp = pd.DataFrame(fighters)
                total_so_far = len(already_scraped) + i + 1
                df_temp.to_csv(
                    f"data/raw/fighters_checkpoint_{total_so_far}.csv",
                    index=False
                )
                print(f"  ✓ Checkpoint saved at {total_so_far} total fighters")

            time.sleep(1.5)

        except Exception as e:
            print(f"  ✗ Failed on {url}: {e}")
            failed.append(url)
            time.sleep(2)
            continue

    driver.quit()

    # ── save final results ────────────────────────────────────────────────────
    print("\n=== STEP 3: Saving final results ===")
    df = pd.DataFrame(fighters)
    df.to_csv("data/raw/all_fighters.csv", index=False)
    print(f"Saved {len(fighters)} fighters to data/raw/all_fighters.csv")

    if failed:
        with open("data/raw/failed_urls.txt", "w") as f:
            for url in failed:
                f.write(url + "\n")
        print(f"{len(failed)} fighters failed — saved to data/raw/failed_urls.txt")

    print("\n=== DONE ===")
    print(f"Total fighters scraped: {len(fighters)}")
    print(f"Total failed: {len(failed)}")


if __name__ == "__main__":
    main()