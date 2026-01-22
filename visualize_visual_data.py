
import pickle
import numpy as np
import os
import cv2
import subprocess
import tempfile
import argparse
from tqdm import tqdm

def visualize_visual_data(data_dir, fps=30, overwrite=False):
    rgbs_file = os.path.join(data_dir, "rgbs_aligned.pkl")
    video_filename = os.path.join(data_dir, "rgb_video.mp4")

    if not os.path.exists(rgbs_file):
        print(f"Error: {rgbs_file} not found.")
        return

    if os.path.exists(video_filename) and not overwrite:
        print(f"Video {video_filename} already exists. Use --overwrite to replace.")
        return

    print(f"Loading {rgbs_file}...")
    with open(rgbs_file, "rb") as f:
        rgbs_data_list = pickle.load(f)

    # Create Video from RGB Images
    if len(rgbs_data_list) > 0:
        print(f"Creating video {video_filename} from {len(rgbs_data_list)} frames...")
        
        first_img = rgbs_data_list[0]
        if isinstance(first_img, np.ndarray):
            # Create a temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Saving frames to temporary directory: {temp_dir}")
                
                # Save each frame as a PNG
                for i, img in enumerate(tqdm(rgbs_data_list, desc="Saving frames")):
                    # collect_realsense.py saves BGR images. 
                    # cv2.imwrite expects BGR. So no conversion needed.
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    cv2.imwrite(frame_path, img)
                
                print("Frames saved. Stitching video with ffmpeg (H.264)...")
                
                try:
                    # Construct ffmpeg command
                    # -framerate: Set input framerate
                    # -i: Input file pattern
                    # -c:v libx264: H.264 codec (VS Code compatible)
                    # -pix_fmt yuv420p: Pixel format for compatibility
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-framerate", str(fps),
                        "-i", os.path.join(temp_dir, "frame_%04d.png"),
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        video_filename
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"Successfully saved video to {video_filename}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"Error creating video with ffmpeg: {e}")
                except FileNotFoundError:
                    print("ffmpeg not found. Falling back to OpenCV (may not play in VS Code)...")
                    height, width = first_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                    for frame in rgbs_data_list:
                        writer.write(frame)
                    writer.release()
                    print(f"Wrote video using OpenCV to {video_filename}")

        else:
            print("RGB data is not numpy array, skipping video creation.")
    else:
        print("No frames found in pickle.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize visual data (RGB video) from pickles.")
    parser.add_argument("--data", default="data/00", help="Path to data directory.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing video.")
    args = parser.parse_args()

    visualize_visual_data(args.data, args.fps, args.overwrite)
