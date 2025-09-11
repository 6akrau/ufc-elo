# src/scrape_roster.py

import requests
from bs4 import BeautifulSoup
import json
import os
from urllib.parse import urljoin
from tqdm import tqdm

# UFCStats character filter base URL
UFCSTATS_BASE = "http://ufcstats.com/statistics/fighters?char={}&page=all"
# User-Agent header for requests
HEADERS = {"User-Agent": (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
)}
# Letters A–Z for filtering
LETTERS = [chr(c) for c in range(ord('A'), ord('Z')+1)]


def scrape_ufcstats():
    """
    Scrape UFCStats A–Z char filter pages for fighter names and profile URLs.
    Each table row represents one fighter; the first column's anchor has the profile link.
    Returns:
        dict: {"Fighter Name": "Profile URL", ...}
    """
    fighters = {}
    for letter in tqdm(LETTERS, desc="Scraping UFCStats"):
        url = UFCSTATS_BASE.format(letter)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
        except requests.RequestException:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        # Iterate rows in the stats table body
        for row in soup.select("tbody tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            # Extract first and last name
            first = cells[0].get_text(strip=True)
            last = cells[1].get_text(strip=True)
            name = f"{first} {last}".strip()
            # The link is in the first cell
            anchor = cells[0].select_one("a[href]")
            if not anchor:
                continue
            href = anchor["href"]
            if name and href.startswith("http"):
                fighters[name] = href
    return fighters


def main():
    # Scrape UFCStats fighters
    print("Scraping UFCStats fighters...")
    fighters = scrape_ufcstats()
    print(f"Found {len(fighters)} fighters on UFCStats.")

    # Save to JSON
    out_dir = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ufc_fighters.json')
    with open(out_path, 'w') as f:
        json.dump(fighters, f, indent=2)

    print(f"Saved fighter list to {out_path}")


if __name__ == '__main__':
    main()
