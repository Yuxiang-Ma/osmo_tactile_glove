#!/usr/bin/env python3
"""
Capture RealSense D435 streams and save data in the same format as the pipeline.

Outputs (under data/<name>/):
  - rgbs_aligned.pkl (list of BGR uint8 images)
  - left_ir_aligned.pkl (list of mono8 uint8 images)
  - right_ir_aligned.pkl (list of mono8 uint8 images)
  - depth_aligned.pkl (list of uint16 depth images, aligned to RGB)
  - rgb_video.mp4 (MP4 video from BGR frames)
"""

import argparse
import os
import pickle
import time
import threading
import queue

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "pyrealsense2 is not installed. Install the RealSense SDK first."
    ) from exc


def _set_option(sensor, option, value, label):
    if not sensor or value is None:
        return
    try:
        if sensor.supports(option):
            sensor.set_option(option, float(value))
        else:
            print(f"Warning: {label} option not supported on this sensor.")
    except Exception as err:
        print(f"Warning: failed to set {label} to {value}: {err}")


def _configure_sensors(profile, auto_exposure, exposure, emitter):
    device = profile.get_device()
    sensors = device.query_sensors()

    color_sensor = None
    depth_sensor = None
    for sensor in sensors:
        name = sensor.get_info(rs.camera_info.name)
        if "RGB Camera" in name:
            color_sensor = sensor
        elif "Stereo Module" in name:
            depth_sensor = sensor

    if color_sensor:
        if auto_exposure is not None:
            _set_option(color_sensor, rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0, "auto_exposure")
        if exposure is not None and not auto_exposure:
            _set_option(color_sensor, rs.option.exposure, exposure, "exposure")
    else:
        print("Warning: RGB sensor not found; skipping exposure settings.")

    if depth_sensor:
        if emitter is not None:
            _set_option(depth_sensor, rs.option.emitter_enabled, 1.0 if emitter else 0.0, "emitter_enabled")
    else:
        print("Warning: Depth sensor not found; skipping emitter setting.")


def _build_output_dir(name):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(repo_root, "data")
    os.makedirs(base_dir, exist_ok=True)
    if name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"realsense_{timestamp}"
    out_dir = os.path.join(base_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _list_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices detected.")
        return 0
    print("Detected RealSense devices:")
    for idx, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        product = dev.get_info(rs.camera_info.product_line)
        fw = dev.get_info(rs.camera_info.firmware_version)
        print(f"  [{idx}] {name} | serial={serial} | product={product} | fw={fw}")
    return len(devices)


def main():
    parser = argparse.ArgumentParser(description="Capture RealSense data and save pkls + mp4.")
    parser.add_argument("--name", default=None, help="Output folder name under data/.")
    parser.add_argument("--width", type=int, default=640, help="Color/IR width.")
    parser.add_argument("--height", type=int, default=480, help="Color/IR height.")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS.")
    parser.add_argument("--duration", type=float, default=None, help="Seconds to record.")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to record.")
    parser.add_argument("--warmup-frames", type=int, default=0, help="Discard initial frames.")
    parser.add_argument("--preview", action="store_true", help="Show live preview window.")
    parser.add_argument("--no-video", action="store_true", help="Skip writing rgb_video.mp4.")
    parser.add_argument("--auto-exposure", action="store_true", help="Enable color auto exposure.")
    parser.add_argument("--exposure", type=float, default=200.0, help="Manual exposure if auto-exposure is off.")
    parser.add_argument("--emitter", action="store_true", help="Enable depth emitter.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--timeout-ms", type=int, default=2000, help="Frame wait timeout in ms.")
    parser.add_argument("--max-timeouts", type=int, default=5, help="Max consecutive timeouts before aborting.")
    parser.add_argument("--serial", default=None, help="Select device by serial number.")
    parser.add_argument("--disable-ir", action="store_true", help="Disable infrared streams to save bandwidth.")
    parser.add_argument("--list-devices", action="store_true", help="List detected RealSense devices and exit.")
    args = parser.parse_args()

    if args.list_devices:
        count = _list_devices()
        raise SystemExit(0 if count > 0 else 1)

    out_dir = _build_output_dir(args.name)
    print(f"Saving data to: {out_dir}")

    output_files = [
        "rgbs_aligned.pkl",
        "depth_aligned.pkl",
    ]
    if not args.disable_ir:
        output_files.append("left_ir_aligned.pkl")
        output_files.append("right_ir_aligned.pkl")

    if not args.no_video:
        output_files.append("rgb_video.mp4")
    if not args.overwrite:
        existing = [f for f in output_files if os.path.exists(os.path.join(out_dir, f))]
        if existing:
            raise SystemExit(f"Output files already exist: {existing}. Use --overwrite to replace.")

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise SystemExit(
            "No RealSense devices detected. Check USB connection/power and udev permissions."
        )

    # Reset device to clear potential stuck states
    print("Resetting RealSense device(s)...")
    reset_count = 0
    for dev in devices:
        try:
            # If serial provided, only reset that specific device
            if args.serial:
                serial = dev.get_info(rs.camera_info.serial_number)
                if serial != args.serial:
                    continue
            
            dev.hardware_reset()
            reset_count += 1
        except Exception as e:
            print(f"Warning: failed to reset device: {e}")
    
    if reset_count > 0:
        # Wait for device to re-enumerate
        print(f"Reset {reset_count} device(s). Waiting 5s for re-enumeration...")
        time.sleep(5)
    else:
        print("No devices reset (serial mismatch?). Proceeding...")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    
    if not args.disable_ir:
        config.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
        config.enable_stream(rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps)

    try:
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        config.resolve(pipeline_wrapper)
    except Exception as err:
        raise SystemExit(
            f"Requested stream profile is not supported: {err}. "
            "Try --width 640 --height 480 --fps 15."
        ) from err

    print("Starting RealSense pipeline...")
    profile = pipeline.start(config)
    _configure_sensors(profile, args.auto_exposure, args.exposure, args.emitter)
    align = rs.align(rs.stream.color)

    # Queues and threading for background saving
    data_queue = queue.Queue()
    save_done_event = threading.Event()

    def save_worker():
        rgbs = []
        left_ir = []
        right_ir = []
        depth = []
        writer = None

        while True:
            item = data_queue.get()
            if item is None:
                break
            
            (color_img, depth_img, left_ir_img, right_ir_img) = item

            # Lazy initialization of writer
            if writer is None and not args.no_video:
                h, w = color_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(os.path.join(out_dir, "rgb_video.mp4"), fourcc, args.fps, (w, h))

            # Save to lists
            rgbs.append(color_img)
            depth.append(depth_img)
            if left_ir_img is not None:
                left_ir.append(left_ir_img)
            if right_ir_img is not None:
                right_ir.append(right_ir_img)

            # Write to video
            if writer is not None:
                writer.write(color_img)

            data_queue.task_done()

        if writer is not None:
            writer.release()
            if not args.no_video:
                print("Saved rgb_video.mp4.")
            else:
                print("Skipped rgb_video.mp4 as requested.")

        print(f"Saving pickle files ({len(rgbs)} frames)...")
        with open(os.path.join(out_dir, "rgbs_aligned.pkl"), "wb") as f:
            pickle.dump(rgbs, f)
        with open(os.path.join(out_dir, "depth_aligned.pkl"), "wb") as f:
            pickle.dump(depth, f)
            
        if not args.disable_ir:
            with open(os.path.join(out_dir, "left_ir_aligned.pkl"), "wb") as f:
                pickle.dump(left_ir, f)
            with open(os.path.join(out_dir, "right_ir_aligned.pkl"), "wb") as f:
                pickle.dump(right_ir, f)
        
        save_done_event.set()

    save_thread = threading.Thread(target=save_worker, daemon=True)
    save_thread.start()

    frame_count = 0
    captured_count = 0
    start_time = time.time()
    timeout_count = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(args.timeout_ms)
            except RuntimeError as err:
                timeout_count += 1
                print(f"Warning: frame wait timed out ({timeout_count}/{args.max_timeouts}): {err}")
                if timeout_count >= args.max_timeouts:
                    raise SystemExit(
                        "Too many timeouts waiting for frames. "
                        "Check USB connection/power and try lower resolution "
                        "(--width 640 --height 480 --fps 15)."
                    ) from err
                continue
            timeout_count = 0
            
            # Align frames to color
            aligned_frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            left_ir_frame = None
            right_ir_frame = None
            if not args.disable_ir:
                left_ir_frame = frames.get_infrared_frame(1)
                right_ir_frame = frames.get_infrared_frame(2)

            # Check validity
            if not color_frame or not depth_frame:
                continue
            if not args.disable_ir and (not left_ir_frame or not right_ir_frame):
                continue

            if frame_count < args.warmup_frames:
                frame_count += 1
                continue

            # Convert to numpy immediately to release RS frame resources
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            
            left_ir_img = None
            right_ir_img = None
            if left_ir_frame:
                left_ir_img = np.asanyarray(left_ir_frame.get_data())
            if right_ir_frame:
                right_ir_img = np.asanyarray(right_ir_frame.get_data())

            # Push to queue
            data_queue.put((color_img, depth_img, left_ir_img, right_ir_img))
            captured_count += 1

            if args.preview:
                cv2.imshow("realsense_rgb", color_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Preview stopped by user.")
                    break

            if args.frames is not None and captured_count >= args.frames:
                break
            if args.duration is not None and (time.time() - start_time) >= args.duration:
                break

    except KeyboardInterrupt:
        print("Capture interrupted by user.")
    finally:
        pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()
        
        # Signal worker to stop
        print("Stopping save worker...")
        data_queue.put(None)
        save_thread.join()
        # Wait for actual save completion (redundant with join but safe)
        save_done_event.wait()

    print(f"Captured {captured_count} frames.")
    print("Done.")


if __name__ == "__main__":
    main()
