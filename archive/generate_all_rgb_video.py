#!/usr/bin/env python3
import pickle
import cv2
import numpy as np

# Load the processed data
print("Loading processed.pkl...")
with open('data/00/processed.pkl', 'rb') as f:
    data = pickle.load(f)

rgb_frames = data['rs_color']
print(f"Total RGB frames: {rgb_frames.shape[0]}")
print(f"Frame shape: {rgb_frames.shape[1:]}")

# Create video writer
output_path = 'data/00/all_211_rgb_frames.mp4'
height, width = rgb_frames.shape[1:3]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Creating video at {output_path}...")
print(f"Resolution: {width}x{height}, FPS: {fps}")

for i, frame in enumerate(rgb_frames):
    # Add frame number text overlay
    frame_with_text = frame.copy()
    cv2.putText(frame_with_text, f'Frame {i}/{len(rgb_frames)}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame_with_text)
    if i % 10 == 0:
        print(f"Processing frame {i}/{len(rgb_frames)}", end='\r')

out.release()
print(f"\n✅ Video saved to {output_path}")
print(f"Total frames written: {len(rgb_frames)}")
print(f"Duration: {len(rgb_frames)/fps:.1f} seconds at {fps} fps")
