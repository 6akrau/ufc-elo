import os
import sys
import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

# Load active fighters map
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ACTIVE_FILE = os.path.join(DATA_DIR, 'active_fighters.json')

with open(ACTIVE_FILE, 'r') as f:
    active_map = json.load(f)

# Create a new driver instance
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
test_driver = webdriver.Chrome(options=chrome_options)

def test_ilia_topuria():
    print("Testing Ilia Topuria ranking extraction...")
    
    # Get the URL
    rw_url = active_map.get('Ilia Topuria', [None, None])[1]
    if not rw_url:
        slug = re.sub(r'[^a-z0-9]', '', 'Ilia Topuria'.lower())
        rw_url = f"https://www.roster.watch/fighters/{slug}.html"
    
    print(f"URL: {rw_url}")
    
    try:
        test_driver.get(rw_url)
        time.sleep(5)  # Wait for JS to load
        html = test_driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        print(f"Page title: {soup.title.string if soup.title else 'No title'}")
        
        # Search for ranking information
        print("\nSearching for ranking info...")
        
        # Method 1: Look for div.rt-text-content (current method)
        elements = soup.select('div.rt-text-content')
        print(f"Found {len(elements)} div.rt-text-content elements")
        
        for i, el in enumerate(elements):
            text = el.get_text(' ', strip=True)
            print(f"Element {i}: {text[:100]}...")
            if 'UFC rank' in text:
                print(f"  *** FOUND UFC RANK: {text} ***")
        
        # Method 2: Search all text for "UFC rank"
        all_text = soup.get_text(' ', strip=True)
        lines = all_text.split('\n')
        for line in lines:
            if 'ufc rank' in line.lower():
                print(f"Found in text: {line}")
        
        # Method 3: Try different selectors
        selectors = [
            'div[class*="rank"]',
            'div[class*="text"]',
            'p',
            'span'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            print(f"\nSelector '{selector}': {len(elements)} elements")
            for el in elements[:3]:
                text = el.get_text(' ', strip=True)
                if 'rank' in text.lower():
                    print(f"  {text[:100]}...")
                    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        test_driver.quit()

if __name__ == "__main__":
    test_ilia_topuria() 