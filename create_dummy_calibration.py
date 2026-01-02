
import numpy as np
import os

# Define a default identity transformation matrix (4x4)
# This represents a camera at the origin with no rotation relative to the base
default_transform = np.eye(4)

# Define the target directory and file path
output_dir = "calibration_data"
output_file = os.path.join(output_dir, "cam_to_base_sea.npy")

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Save the uniform transformation matrix to the file
np.save(output_file, default_transform)

print(f"Successfully created dummy calibration file at: {output_file}")
print("Transformation matrix:")
print(default_transform)
