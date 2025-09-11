# src/combine_fighters.py

import os
import json
from tqdm import tqdm

# Data directory and file paths
data_dir = os.path.join(os.path.dirname(__file__), '../data')
ufc_file = os.path.join(data_dir, 'ufc_fighters.json')
rw_file = os.path.join(data_dir, 'rosterwatch_fighters.json')
output_file = os.path.join(data_dir, 'active_fighters.json')

# Ensure data directory exists
os.makedirs(data_dir, exist_ok=True)

# Load UFCStats data
with open(ufc_file, 'r') as f:
    ufc_map = json.load(f)  # dict: name -> ufcstats_url

# Load roster.watch data
with open(rw_file, 'r') as f:
    rw_map = json.load(f)   # dict: name -> rosterwatch_url

# Combine entries present in both maps
combined = {}
for name, ufc_url in tqdm(ufc_map.items(), desc='Combining fighters'):
    rw_url = rw_map.get(name)
    if rw_url:
        # Map name to [ufcstats_url, rosterwatch_url]
        combined[name] = [ufc_url, rw_url]

# Write combined JSON
with open(output_file, 'w') as f:
    json.dump(combined, f, indent=2)

print(f"Saved {len(combined)} active fighters to {output_file}")
