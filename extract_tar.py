import tarfile
import os

# Assuming running from project root
tar_path = "hamer/_DATA/hamer_demo_data.tar.gz"
extract_path = "hamer" 

if os.path.exists(tar_path):
    print(f"Extracting {tar_path} to {extract_path}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"File not found: {tar_path}")
