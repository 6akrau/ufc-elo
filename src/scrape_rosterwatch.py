# src/scrape_rosterwatch.py

import os
import json
import time
from urllib.parse import urljoin
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Data directory and output file
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'rosterwatch_fighters.json')
os.makedirs(DATA_DIR, exist_ok=True)

# Base URL for roster.watch profile links
PROFILE_BASE = 'https://www.roster.watch'
# Weight class pages to scrape
WEIGHT_CLASSES = [
    'heavyweight',
    'lightheavyweight',
    'middleweight',
    'welterweight',
    'lightweight',
    'featherweight',
    'bantamweight',
    'flyweight',
]

# Setup headless Chrome driver
def init_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    return webdriver.Chrome(options=options)


def scrape_rosterwatch():
    """
    Use Selenium to render each weight-class page on roster.watch and
    scrape fighter profile links. Returns a dict mapping fighter name to URL.
    """
    driver = init_driver()
    fighters = {}

    for cls in tqdm(WEIGHT_CLASSES, desc='Scraping weight classes'):
        page_url = f"{PROFILE_BASE}/{cls}.html"
        try:
            driver.get(page_url)
            # allow JS to render
            time.sleep(3)
        except Exception:
            continue
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # Extract profile links on the weight class page
        for a in soup.select('a[href^="/fighters/"]'):
            href = a['href'].strip()
            name = a.get_text(strip=True)
            if not name or not href.endswith('.html'):
                continue
            full_url = urljoin(PROFILE_BASE, href)
            fighters[name] = full_url

    driver.quit()
    return fighters


def main():
    print('Scraping roster.watch fighters by weight class via Selenium...')
    roster_map = scrape_rosterwatch()
    print(f'Found {len(roster_map)} roster.watch fighters')

    # Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(roster_map, f, indent=2)
    print(f'Saved roster.watch fighters to {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
