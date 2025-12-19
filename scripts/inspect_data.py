import rioxarray
import numpy as np
import glob
import os

# Grab the first available Landsat file
files = glob.glob("data/raw_samples/pair_*/*landsat_B4.tif")
if not files:
    print("No Landsat files found!")
    exit()

filename = files[0]
print(f"Inspecting: {filename}")

# Open and analyze
da = rioxarray.open_rasterio(filename)
data = da.values.flatten()

# Filter out 0 (NoData) to see the real numbers
valid_data = data[data != 0]

print(f"--- STATISTICS ---")
print(f"Min Value: {np.min(valid_data)}")
print(f"Max Value: {np.max(valid_data)}")
print(f"Mean Value: {np.mean(valid_data)}")
print(f"Data Type: {da.dtype}")
