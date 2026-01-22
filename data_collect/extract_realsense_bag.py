#!/usr/bin/env python3
"""
Extract RealSense .bag into rgb/depth videos and pipeline-compatible pickle files.

Outputs (under data/<bag_stem>/ unless --out-dir is provided):
  - rgbs_aligned.pkl
  - left_ir_aligned.pkl
  - right_ir_aligned.pkl
  - depth_aligned.pkl
  - rgb_video.mp4
  - depth_video.mp4 (colorized)
"""

import argparse
import os
import pickle

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "pyrealsense2 is not installed. Install the RealSense SDK first."
    ) from exc


def _resolve_out_dir(bag_path, out_dir):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if out_dir is None:
        bag_stem = os.path.splitext(os.path.basename(bag_path))[0]
        out_dir = os.path.join(repo_root, "data", bag_stem)
    elif not os.path.isabs(out_dir):
        out_dir = os.path.join(repo_root, "data", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _format_name(fmt):
    try:
        return str(fmt).split(".")[-1]
    except Exception:
        return str(fmt)


def main():
    parser = argparse.ArgumentParser(description="Extract RealSense .bag to videos and pkls.")
    parser.add_argument("--bag", required=True, help="Path to .bag file.")
    parser.add_argument("--out-dir", default=None, help="Output dir or name under data/.")
    parser.add_argument("--video-fps", type=int, default=30, help="FPS for output videos.")
    parser.add_argument("--depth-scale", type=float, default=0.03, help="Scale factor for depth colorization.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to extract.")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame.")
    parser.add_argument("--no-align", action="store_true", help="Disable depth-to-color alignment.")
    parser.add_argument("--no-video", action="store_true", help="Skip writing mp4 videos.")
    parser.add_argument("--no-pkl", action="store_true", help="Skip writing pickle files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    bag_path = os.path.abspath(args.bag)
    if not os.path.exists(bag_path):
        raise SystemExit(f"Bag file not found: {bag_path}")

    out_dir = _resolve_out_dir(bag_path, args.out_dir)
    print(f"Saving outputs to: {out_dir}")

    output_files = [
        "rgbs_aligned.pkl",
        "left_ir_aligned.pkl",
        "right_ir_aligned.pkl",
        "depth_aligned.pkl",
        "rgb_video.mp4",
        "depth_video.mp4",
    ]
    if not args.overwrite:
        existing = [f for f in output_files if os.path.exists(os.path.join(out_dir, f))]
        if existing:
            raise SystemExit(f"Output files already exist: {existing}. Use --overwrite to replace.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)

    print("Starting playback...")
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    try:
        playback.set_real_time(False)
    except Exception as err:
        print(f"Warning: unable to set non-real-time playback: {err}")

    streams = profile.get_streams()
    stream_summary = []
    for stream in streams:
        stream_summary.append(
            f"{stream.stream_type().name}:{stream.stream_index()} "
            f"{stream.format().name} {stream.as_video_stream_profile().width()}x{stream.as_video_stream_profile().height()}"
        )
    print("Available streams:", ", ".join(stream_summary))

    align = rs.align(rs.stream.color) if not args.no_align else None
    align_disabled = args.no_align

    rgbs = []
    left_ir = []
    right_ir = []
    depth = []
    timestamps = []

    rgb_writer = None
    depth_writer = None

    frame_idx = 0
    saved_idx = 0
    missing_depth = 0
    missing_ir1 = 0
    missing_ir2 = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(5000)
            except RuntimeError:
                if playback:
                    try:
                        if playback.current_status() == rs.playback_status.stopped:
                            break
                    except Exception:
                        pass
                continue

            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            aligned_frames = None
            if align and not align_disabled:
                try:
                    aligned_frames = align.process(frames)
                except RuntimeError as err:
                    align_disabled = True
                    print(f"Warning: depth-to-color alignment failed, using unaligned depth. Error: {err}")

            color_frame = frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame() if aligned_frames else frames.get_depth_frame()
            left_ir_frame = frames.get_infrared_frame(1)
            right_ir_frame = frames.get_infrared_frame(2)

            if not color_frame:
                frame_idx += 1
                continue

            color_img = np.asanyarray(color_frame.get_data())
            if color_frame.get_profile().format() == rs.format.rgb8:
                color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            depth_img = None
            if depth_frame:
                depth_img = np.asanyarray(depth_frame.get_data())
            else:
                missing_depth += 1

            left_ir_img = None
            if left_ir_frame:
                left_ir_img = np.asanyarray(left_ir_frame.get_data())
            else:
                missing_ir1 += 1

            right_ir_img = None
            if right_ir_frame:
                right_ir_img = np.asanyarray(right_ir_frame.get_data())
            else:
                missing_ir2 += 1

            if rgb_writer is None and not args.no_video:
                height, width = color_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                rgb_writer = cv2.VideoWriter(os.path.join(out_dir, "rgb_video.mp4"), fourcc, args.video_fps, (width, height))

            if depth_img is not None and depth_writer is None and not args.no_video:
                height, width = depth_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                depth_writer = cv2.VideoWriter(os.path.join(out_dir, "depth_video.mp4"), fourcc, args.video_fps, (width, height))

            rgbs.append(color_img)
            left_ir.append(left_ir_img)
            right_ir.append(right_ir_img)
            depth.append(depth_img)
            timestamps.append(color_frame.get_timestamp())
            saved_idx += 1

            if rgb_writer is not None:
                rgb_writer.write(color_img)

            if depth_img is not None and depth_writer is not None:
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=args.depth_scale),
                    cv2.COLORMAP_JET,
                )
                depth_writer.write(depth_vis)

            frame_idx += 1
            if args.max_frames is not None and saved_idx >= args.max_frames:
                break

    finally:
        pipeline.stop()
        if rgb_writer is not None:
            rgb_writer.release()
        if depth_writer is not None:
            depth_writer.release()

    print(f"Saved frames: {saved_idx}")
    if missing_depth:
        print(f"Warning: {missing_depth} frames missing depth.")
    if missing_ir1:
        print(f"Warning: {missing_ir1} frames missing left IR.")
    if missing_ir2:
        print(f"Warning: {missing_ir2} frames missing right IR.")

    if not args.no_pkl:
        with open(os.path.join(out_dir, "rgbs_aligned.pkl"), "wb") as f:
            pickle.dump(rgbs, f)
        with open(os.path.join(out_dir, "left_ir_aligned.pkl"), "wb") as f:
            pickle.dump(left_ir, f)
        with open(os.path.join(out_dir, "right_ir_aligned.pkl"), "wb") as f:
            pickle.dump(right_ir, f)
        with open(os.path.join(out_dir, "depth_aligned.pkl"), "wb") as f:
            pickle.dump(depth, f)
        print("Wrote pickle files.")
    else:
        print("Skipped pickle files.")

    if args.no_video:
        print("Skipped videos.")


if __name__ == "__main__":
    main()
