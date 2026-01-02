
import pickle
import numpy as np
import os

data_dir = "data/00"
mags_file = os.path.join(data_dir, "synced_mags_aligned.pkl")
rgbs_file = os.path.join(data_dir, "rgbs_aligned.pkl")

print(f"Inspecting {mags_file}")
try:
    with open(mags_file, "rb") as f:
        mags_data = pickle.load(f)
    print(f"Mags data type: {type(mags_data)}")
    if isinstance(mags_data, np.ndarray):
        print(f"Mags data shape: {mags_data.shape}")
        if mags_data.size > 0:
            print(f"Sample mags data (first element): {mags_data[0]}")
    elif isinstance(mags_data, list):
        print(f"Mags data list length: {len(mags_data)}")
        if len(mags_data) > 0:
            print(f"Sample mags data (first element): {mags_data[0]}")
except Exception as e:
    print(f"Error loading mags data: {e}")

print("-" * 20)

print(f"Inspecting {rgbs_file}")
try:
    with open(rgbs_file, "rb") as f:
        rgbs_data = pickle.load(f)
    print(f"Rgbs data type: {type(rgbs_data)}")
    if isinstance(rgbs_data, np.ndarray):
        print(f"Rgbs data shape: {rgbs_data.shape}")
    elif isinstance(rgbs_data, list):
        print(f"Rgbs data list length: {len(rgbs_data)}")
except Exception as e:
    print(f"Error loading rgbs data: {e}")
