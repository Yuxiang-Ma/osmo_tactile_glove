#!/usr/bin/env python3
"""
Create rgb_video.mp4 from rgbs_aligned.pkl in a data/<name>/ folder.
Uses ffmpeg (if available) for H.264 encoding compatible with VS Code/Web.
Falls back to cv2.VideoWriter (mp4v) if ffmpeg is missing.
"""

import argparse
import os
import pickle
import subprocess
import tempfile
import shutil
import sys

import cv2


def _resolve_data_dir(name_or_path):
    if os.path.isabs(name_or_path):
        return name_or_path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", name_or_path)


def main():
    parser = argparse.ArgumentParser(description="Create rgb_video.mp4 from rgbs_aligned.pkl.")
    parser.add_argument("--data", required=True, help="Data folder name under data/ or absolute path.")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (negative means till end).")
    parser.add_argument("--output", default="rgb_video.mp4", help="Output video filename.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    args = parser.parse_args()

    data_dir = _resolve_data_dir(args.data)
    rgb_path = os.path.join(data_dir, "rgbs_aligned.pkl")
    if not os.path.exists(rgb_path):
        raise SystemExit(f"Missing {rgb_path}")

    out_path = os.path.join(data_dir, args.output)
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"{out_path} exists. Use --overwrite to replace.")

    print(f"Loading {rgb_path}...")
    with open(rgb_path, "rb") as f:
        rgbs = pickle.load(f)

    if args.end == -1:
        frames = rgbs[args.start:]
    else:
        frames = rgbs[args.start:args.end]

    if not frames:
        raise SystemExit("No frames found for the given range.")

    height, width = frames[0].shape[:2]
    
    # Try ffmpeg first
    ffmpeg_success = False
    try:
        # Check if ffmpeg is installed
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"Encoding {len(frames)} frames with ffmpeg (H.264)...")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save frames to temp dir
            for i, frame in enumerate(frames):
                # frame is BGR. cv2.imwrite expects BGR.
                filename = os.path.join(temp_dir, f"frame_{i:04d}.png")
                cv2.imwrite(filename, frame)
            
            # Run ffmpeg
            # -y: overwrite output
            # -framerate: input fps
            # -i: input pattern
            # -c:v libx264: codec
            # -pix_fmt yuv420p: pixel format for wide compatibility
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(args.fps),
                "-i", os.path.join(temp_dir, "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                out_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ffmpeg_success = True
            print(f"Successfully saved video to {out_path}")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ffmpeg failed or not found ({e}). Falling back to cv2.VideoWriter...")
    
    if not ffmpeg_success:
        print("Using OpenCV VideoWriter (mp4v). Note: This format may not play in VS Code.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, args.fps, (width, height))

        for frame in frames:
            writer.write(frame)
        writer.release()
        print(f"Wrote {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
