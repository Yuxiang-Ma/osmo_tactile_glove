
import pickle
import numpy as np
import os

path = "data/00/"
file_path = path + "synced_mags_aligned.pkl"

print(f"Loading {file_path}")
loaded_data = pickle.load(open(file_path, "rb"))
print(f"Loaded type: {type(loaded_data)}")

if isinstance(loaded_data, np.ndarray):
    print(f"Loaded shape: {loaded_data.shape}")
    print(f"Loaded dtype: {loaded_data.dtype}")

bowie_data = np.asarray(loaded_data, dtype="object")
print(f"After asarray(dtype=object): shape={bowie_data.shape}, ndim={bowie_data.ndim}")

try:
    sliced = bowie_data[:,:,1:]
    print(f"Slicing [:,:,1:] successful. Result shape: {sliced.shape}")
except Exception as e:
    print(f"Slicing failed: {e}")

# Simulate the logic in extract_hamer.py
if bowie_data.ndim == 2 and bowie_data.shape[1] == 2:
    print("Detected compact format (ndim=2)")
else:
    print("Not compact format")
    try:
        reshaped = bowie_data.reshape((bowie_data.shape[0], 10, 4))
        print(f"Reshape successful: {reshaped.shape}")
        sliced_final = reshaped[:,:,1:]
        print(f"Final slice successful: {sliced_final.shape}")
    except Exception as e:
        print(f"Reshape/Slice failed: {e}")
