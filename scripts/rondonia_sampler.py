import os
import datetime
from pystac_client import Client
import planetary_computer
import requests
from tqdm import tqdm

# --- CONFIGURATION ---
AOI_POINT = {"lon": -61.9, "lat": -10.8} 
SEARCH_START = "2023-06-01"
SEARCH_END = "2023-08-30"

# This will now automatically go to your External Drive via the shortcut
OUTPUT_DIR = "data/raw_samples"
MAX_PAIRS = 5

def get_catalog():
    return Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

def download_asset(asset_href, save_path):
    if os.path.exists(save_path):
        print(f"   [Skip] File exists: {os.path.basename(save_path)}")
        return

    try:
        r = requests.get(asset_href, stream=True)
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        if total_size < 10000: return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(save_path), leave=False) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except Exception as e:
        print(f"   [Error] Failed to download: {e}")

def main():
    catalog = get_catalog()
    print(f"--- Rondonia Micro-Sampler (External Storage) ---")
    
    print("1. Searching for clear Sentinel-2 scenes...")
    search_s2 = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [AOI_POINT["lon"], AOI_POINT["lat"]]},
        datetime=f"{SEARCH_START}/{SEARCH_END}",
        query={"eo:cloud_cover": {"lt": 10}}
    )
    s2_items = list(search_s2.items())
    print(f"   Found {len(s2_items)} clear Sentinel scenes.")

    pairs_found = 0
    
    for s2_item in s2_items:
        if pairs_found >= MAX_PAIRS: break
            
        s2_date = s2_item.datetime
        print(f"\nChecking match for Sentinel date: {s2_date.date()}...")

        start_window = (s2_date - datetime.timedelta(days=3)).strftime("%Y-%m-%d")
        end_window = (s2_date + datetime.timedelta(days=3)).strftime("%Y-%m-%d")

        search_l8 = catalog.search(
            collections=["landsat-c2-l2"],
            intersects={"type": "Point", "coordinates": [AOI_POINT["lon"], AOI_POINT["lat"]]},
            datetime=f"{start_window}/{end_window}",
            query={"eo:cloud_cover": {"lt": 20}}
        )
        l8_items = list(search_l8.items())

        if not l8_items:
            print("   No matching Landsat scene found. Skipping.")
            continue
        
        l8_item = sorted(l8_items, key=lambda x: x.properties["eo:cloud_cover"])[0]
        print(f"   MATCH FOUND! Landsat ID: {l8_item.id}")
        pairs_found += 1
        
        pair_dir = os.path.join(OUTPUT_DIR, f"pair_{pairs_found}_{s2_date.date()}")
        
        print("   Downloading Sentinel-2 assets...")
        for band in ["B02", "B03", "B04", "B08"]: 
            if band in s2_item.assets:
                download_asset(s2_item.assets[band].href, os.path.join(pair_dir, f"sentinel_{band}.tif"))

        print("   Downloading Landsat assets...")
        l8_bands = {"blue": "B2", "green": "B3", "red": "B4", "nir08": "B5"}
        for key, name in l8_bands.items():
            if key in l8_item.assets:
                download_asset(l8_item.assets[key].href, os.path.join(pair_dir, f"landsat_{name}.tif"))

    print(f"\n--- Done! Data saved to External Drive via: {os.path.abspath(OUTPUT_DIR)} ---")

if __name__ == "__main__":
    main()
