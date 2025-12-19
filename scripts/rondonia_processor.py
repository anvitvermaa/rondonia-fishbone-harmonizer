import os
import glob
import numpy as np
import rioxarray
import xarray as xr
from tqdm import tqdm
from skimage.io import imsave
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
INPUT_DIR = "data/raw_samples"
OUTPUT_DIR = "data/ready_to_train"
CHIP_SIZE = 256
S2_BANDS = ["B04", "B03", "B02"]
L8_BANDS = ["B4", "B3", "B2"] 

def normalize_smart(data):
    """
    Robust normalization using percentiles.
    This fixes the 'Solid White' issue by automatically finding 
    the min/max range of the actual data.
    """
    # 1. Flatten to find stats, ignoring pure black (0) background
    valid_pixels = data[data > 0]
    
    # If the image is empty or has too few valid pixels, return None
    if valid_pixels.size < 100:
        return None
        
    # 2. Find the 2nd and 98th percentile (Safe Min/Max)
    p2, p98 = np.percentile(valid_pixels, (2, 98))
    
    # 3. Stretch the data to 0-1 range
    # Avoid division by zero
    if p98 == p2:
        return np.zeros_like(data, dtype=np.uint8)
        
    scaled = np.clip((data - p2) / (p98 - p2), 0, 1)
    
    # 4. Convert to 0-255
    return (scaled * 255).astype(np.uint8)

def load_and_stack(folder, platform, band_list):
    layers = []
    for band in band_list:
        search_path = os.path.join(folder, f"{platform}_{band}.tif")
        files = glob.glob(search_path)
        if not files:
            raise FileNotFoundError(f"Missing {band} in {folder}")
        da = rioxarray.open_rasterio(files[0])
        layers.append(da)
    stack = xr.concat(layers, dim="band")
    stack.coords["band"] = band_list
    return stack

def process_pair(pair_folder, pair_id):
    try:
        s2_stack = load_and_stack(pair_folder, "sentinel", S2_BANDS)
        l8_stack = load_and_stack(pair_folder, "landsat", L8_BANDS)
        l8_aligned = l8_stack.rio.reproject_match(s2_stack)

        width = s2_stack.rio.width
        height = s2_stack.rio.height
        chips_created = 0
        
        for y in range(0, height - CHIP_SIZE, CHIP_SIZE):
            for x in range(0, width - CHIP_SIZE, CHIP_SIZE):
                
                s2_chip = s2_stack.isel(x=slice(x, x+CHIP_SIZE), y=slice(y, y+CHIP_SIZE))
                l8_chip = l8_aligned.isel(x=slice(x, x+CHIP_SIZE), y=slice(y, y+CHIP_SIZE))

                s2_data = s2_chip.values.transpose(1, 2, 0)
                l8_data = l8_chip.values.transpose(1, 2, 0)

                # QUALITY CHECK:
                # If more than 10% of the image is empty black space (NoData), skip it.
                # This fixes the "Jagged Edge" images.
                if np.count_nonzero(s2_data) / s2_data.size < 0.9:
                    continue

                # NORMALIZE (The Fix)
                s2_img = normalize_smart(s2_data)
                l8_img = normalize_smart(l8_data)

                if s2_img is None or l8_img is None:
                    continue

                filename = f"{pair_id}_{x}_{y}.png"
                imsave(os.path.join(OUTPUT_DIR, "HR", filename), s2_img)
                imsave(os.path.join(OUTPUT_DIR, "LR", filename), l8_img)
                chips_created += 1

        return chips_created

    except Exception as e:
        print(f"Error processing {pair_folder}: {e}")
        return 0

def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "HR"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "LR"), exist_ok=True)
    
    pair_folders = glob.glob(os.path.join(INPUT_DIR, "pair_*"))
    print(f"--- Starting SMART Processing Factory ---")
    
    total_chips = 0
    for folder in tqdm(pair_folders):
        pair_id = os.path.basename(folder)
        count = process_pair(folder, pair_id)
        total_chips += count
        
    print(f"\n--- Processing Complete ---")
    print(f"Generated {total_chips} valid training chips.")

if __name__ == "__main__":
    main()
