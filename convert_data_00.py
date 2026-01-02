
import pickle
import numpy as np
import os

def convert_bowie_data(input_path, output_path):
    print(f"Loading {input_path}...")
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    
    data = np.asarray(data, dtype=object)
    
    # Check if it matches the compact format: (N, 2) where col 1 is a list/array of 30 floats
    if data.ndim == 2 and data.shape[1] == 2:
        sample_val = data[0, 1]
        if (isinstance(sample_val, list) or isinstance(sample_val, np.ndarray)) and len(sample_val) == 30:
            print("Detected compact format (N, 2). Converting to (N, 10, 4)...")
            
            new_data = []
            for i in range(len(data)):
                ts = data[i, 0]
                vals = np.array(data[i, 1])
                
                # Reshape 30 values to (10 sensors, 3 axes)
                vals_reshaped = vals.reshape(10, 3)
                
                # Create (10, 4) block: [timestamp, x, y, z]
                block = np.zeros((10, 4))
                block[:, 0] = ts
                block[:, 1:] = vals_reshaped
                
                new_data.append(block)
            
            converted_data = np.array(new_data)
            print(f"Conversion complete. New shape: {converted_data.shape}")
            
            print(f"Saving to {output_path}...")
            with open(output_path, "wb") as f:
                pickle.dump(converted_data, f)
            print("Done.")
        else:
            print("Data is (N, 2) but second column format is unexpected. Skipping conversion.")
            print(f"Sample second column: {type(sample_val)}, length {len(sample_val) if hasattr(sample_val, '__len__') else 'N/A'}")
    else:
        print(f"Data shape is {data.shape}, expected (N, 2) for compact format. Skipping conversion.")

if __name__ == "__main__":
    input_file = "data/00/synced_mags_aligned.pkl"
    output_file = "data/00/synced_mags_aligned_fixed.pkl"
    
    if os.path.exists(input_file):
        convert_bowie_data(input_file, output_file)
    else:
        print(f"Input file {input_file} not found.")
