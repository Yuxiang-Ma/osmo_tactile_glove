
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def visualize_data(data_dir):
    mags_file = os.path.join(data_dir, "synced_mags_aligned.pkl")
    rgbs_file = os.path.join(data_dir, "rgbs_aligned.pkl")

    print(f"Loading {mags_file}...")
    with open(mags_file, "rb") as f:
        mags_data_list = pickle.load(f)

    print(f"Loading {rgbs_file}...")
    with open(rgbs_file, "rb") as f:
        rgbs_data_list = pickle.load(f)

    # Process mags data
    timestamps = []
    sensor_data = []

    for item in mags_data_list:
        # Item is likely (timestamp, list_of_30_floats)
        ts = item[0]
        values = item[1]
        timestamps.append(ts)
        # Reshape values to (10, 3) -> 10 sensors, 3 axes
        if isinstance(values, list):
            values = np.array(values)
        values = values.reshape(10, 3)
        sensor_data.append(values)

    sensor_data = np.array(sensor_data) # Shape: (N, 10, 3)
    timestamps = np.array(timestamps)

    print(f"Processed sensor data shape: {sensor_data.shape}")

    # Plotting Magnetometer Data
    num_sensors = sensor_data.shape[1]
    fig, axes = plt.subplots(num_sensors, 1, figsize=(10, 2 * num_sensors), sharex=True)
    
    if num_sensors == 1:
        axes = [axes]

    for i in range(num_sensors):
        ax = axes[i]
        ax.plot(sensor_data[:, i, 0], label='X', color='r')
        ax.plot(sensor_data[:, i, 1], label='Y', color='g')
        ax.plot(sensor_data[:, i, 2], label='Z', color='b')
        ax.set_ylabel(f'Sensor {i}')
        if i == 0:
            ax.legend(loc='upper right')
        if i == num_sensors - 1:
            ax.set_xlabel('Frame Index')

    plt.suptitle(f'Magnetometer Data for {data_dir}')
    plt.tight_layout()
    plot_filename = os.path.join(data_dir, "mags_plot.png")
    plt.savefig(plot_filename)
    print(f"Saved plot to {plot_filename}")
    plt.close()

    # Create Video from RGB Images
    if len(rgbs_data_list) > 0:
        video_filename = os.path.join(data_dir, "rgb_video.mp4")
        print(f"Creating video {video_filename}...")
        
        # Check first image to get dimensions
        first_img = rgbs_data_list[0]
        if isinstance(first_img, np.ndarray):
             height, width, layers = first_img.shape
             size = (width, height)
             
             # Initialize video writer
             out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
             
             for i, img in enumerate(rgbs_data_list):
                 # Convert RGB to BGR for OpenCV
                 img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                 out.write(img_bgr)
             
             out.release()
             print(f"Saved video to {video_filename}")
        else:
            print("RGB data is not numpy array, skipping video creation.")

if __name__ == "__main__":
    visualize_data("data/00")
