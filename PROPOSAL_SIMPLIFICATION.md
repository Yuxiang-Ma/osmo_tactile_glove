# Codebase Simplification Proposal: Visual-Only Pipeline

## Objective
Remove dependencies on magnetic sensor ("Bowie") data and hardware to streamline the codebase for a visual-only hand pose extraction pipeline.

## 1. Data Collection & Processing
The core visual pipeline is already largely independent. We will formalize this by removing magnetic data expectations.

*   **`data_collect/collect_realsense.py`**:
    *   **Status:** Already independent.
    *   **Action:** No changes needed (already supports `--disable-ir` and doesn't check for Bowie hardware).

*   **`labs/glove2robot/postprocess/extract_hamer.py`**:
    *   **Status:** Safe. The `GlovePoseTracker.open_sync_pkl` method loads RGB/Depth/IR but does *not* load `synced_mags_aligned.pkl`.
    *   **Action:** None required.

*   **`scripts/bowie/convert_bag_to_pkl.py` (Legacy Support)**:
    *   **Status:** Currently expects `/bowie/synced` topic.
    *   **Action:** Modify to make the Bowie topic optional. If missing, it should still extract RGB/Depth/IR to pickle files.

## 2. Visualization Tools
Current visualization scripts are tightly coupled with magnetic data plotting.

*   **`visualize_00.py`**:
    *   **Status:** Fails if `synced_mags_aligned.pkl` is missing.
    *   **Action:** Create a new script `visualize_visual_data.py` (or refactor existing) that:
        *   Checks for `rgbs_aligned.pkl`.
        *   Generates the RGB video (using `ffmpeg` for VS Code compatibility).
        *   Skips all magnetometer plotting logic.

## 3. Cleanup & Archival
Remove or move code related to the magnetic glove hardware to avoid confusion.

*   **Move to `archive/` (or delete):**
    *   `firmware/` (STM32 code for the glove).
    *   `hardware/ros2/bowie_ros2_ws/` (ROS 2 driver for the glove).
    *   `convert_data_00.py` (Specific to Bowie data format conversion).
    *   `debug_bowie_loading.py`
    *   `labs/glove2robot/utils/bowie.py`
    *   `labs/glove2robot/utils/bowie_data.py`
    *   `labs/glove2robot/utils/bowiepb/`
    *   `scripts/plot_keypoints_with_osmo.py` (Heavily dependent on Bowie data).

## 4. Dependencies
*   **`labs/glove2robot/utils/glove_utils.py`**:
    *   **Action:** This imports `BowieGlove`. If other scripts import `glove_utils`, they might break. We should stub out or remove the Bowie parts of this utility file.

## Execution Plan
1.  **Refactor Visualization:** Create `visualize_visual_data.py` immediately to allow verifying data without magnetic sensors.
2.  **Archive Hardware Code:** Move `firmware` and `hardware` folders to `archive/`.
3.  **Clean Utils:** Audit `labs/glove2robot/utils` to remove unused Bowie imports.
