# Extracting Hand pose tracking from OSMO

## Todo
- [ ] extract hamer pose tracking code and retargetting code only
- [ ] update conda env requirements
- [ ] Test the code on our hardware
- [ ] A more generalizable way of retargetting hand poses
- [ ] Integrate with popular and simple hand simulation environment

 
## Problems
### osmo environment bugs
The osmo python env has complex dependency settings, packages like hamer, sam, detectron2 can not be installed simply with conda. This is what gemini did to install everything.

### dataset 



### Data Format & Shape Inconsistency
The `extract_hamer.py` script expects magnetometer data in `synced_mags_aligned.pkl` to be reshape-able to `(N, 10, 4)` (Time, 10 Sensors, [Timestamp, X, Y, Z]). However, some data versions (like `data/00/`) use a compact format: `(N, 2)` where the second column is a list of 30 floats (10 sensors * 3 axes).

You can inspect the data with inspect_pkl.py.

**Fix Implemented:**
`labs/glove2robot/postprocess/extract_hamer.py` now automatically detects this compact format and converts it on-the-fly:
1.  Checks if data shape is `(N, 2)` and column 1 contains a list of 30 elements.
2.  Extracts the timestamp from column 0.
3.  Reshapes the 30-float list into `(10, 3)` (X, Y, Z for 10 sensors).
4.  Constructs the expected `(10, 4)` block by prepending the timestamp to each sensor row.

This ensures compatibility with both old and new data formats without requiring manual file conversion.

### Installation Steps

1.  **Create Conda Environment**:
    ```bash
    conda create -n osmo python=3.10
    conda activate osmo
    ```

2.  **Install PyTorch (Specific Version)**:
    It is crucial to use PyTorch 2.4.1 with CUDA 12.1 compatibility to match `flash-attn` and `sam-2` requirements.
    ```bash
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

3.  **Install Flash Attention**:
    Installed via pip. Verified version `2.7.4.post1` works with the above torch version.
    ```bash
    pip install flash-attn==2.7.4.post1
    ```

4.  **Install Detectron2**:
    Must be installed from source to ensure CUDA extensions are built against the correct PyTorch version.
    ```bash
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```

5.  **Install HaMeR (Hand Mesh Recovery)**:
    Cloned and installed from source.
    ```bash
    git clone https://github.com/geopavlakos/hamer.git
    cd hamer
    pip install -e .[all]
    pip install -v -e third-party/ViTPose
    cd ..
    ```

6.  **Install SAM 2 (Segment Anything 2)**:
    Cloned and installed from source. Note: SAM 2 setup might request `torch>=2.5.1` but we verified it works with `2.4.1`.
    ```bash
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    pip install -e .
    cd ..
    ```

7.  **Install MMPose & Open3D**:
    MMPose requires `openmim`.
    ```bash
    pip install openmim
    mim install mmengine
    mim install "mmcv>=2.0.0"
    mim install "mmpose==0.24.0"
    pip install open3d
    ```

8.  **Fix NumPy Version**:
    NumPy 2.x causes `AttributeError: _ARRAY_API` and binary incompatibility with some compiled wheels (like `flash-attn` or `detectron2`). We downgraded to 1.x.
    ```bash
    pip install "numpy<2.0"  # Installed 1.26.4
    ```

9.  **Project Path Setup (glove2robot & glovedp)**:
    To allow imports like `import glove2robot` and `import glovedp` without modifying `sys.path` in every script, we added a `.pth` file to the environment's site-packages.
    - File created: `/home/yxma/miniconda3/envs/osmo/lib/python3.10/site-packages/osmo_local.pth`
    - Content:
      ```
      /home/yxma/src/osmo_tactile_glove
      /home/yxma/src/osmo_tactile_glove/labs
      ```

10. **FoundationStereo Alias**:
    The code expects `mmint_foundationstereo` but the package is often just `FoundationStereo`. We created a symbolic link in site-packages to map the import.
    ```bash
    ln -s /home/yxma/src/osmo_tactile_glove/FoundationStereo /home/yxma/miniconda3/envs/osmo/lib/python3.10/site-packages/mmint_foundationstereo
    ```

### Detailed Setup & Troubleshooting Summary (Replication Guide)

If replicating this setup on a new machine, follow these additional steps to resolve common errors encountered during the initial setup:

1.  **HaMeR Recursive Clone & ViTPose**:
    The `hamer` repository must be cloned recursively to include `ViTPose`. If `hamer.vitpose_model` is missing:
    ```bash
    # If not cloned recursively initially:
    git clone --recursive https://github.com/geopavlakos/hamer.git hamer
    # Install ViTPose
    pip install -v -e hamer/third-party/ViTPose
    # Re-install HaMeR
    pip install -e hamer
    ```
    *Fix Applied:* We also moved `hamer/vitpose_model.py` into `hamer/hamer/vitpose_model.py` to fix import errors (`ModuleNotFoundError: No module named 'hamer.vitpose_model'`).

2.  **MANO Models**:
    Proprietary MANO hand models are required but not included. Download `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` and place them in `hamer/_DATA/data/mano/`.
    ```bash
    mkdir -p hamer/_DATA/data/mano
    wget -O hamer/_DATA/data/mano/MANO_RIGHT.pkl https://github.com/huchenlei/HandRefinerPortable/raw/master/mesh_graphormer/modeling/data/MANO_RIGHT.pkl
    wget -O hamer/_DATA/data/mano/MANO_LEFT.pkl https://github.com/huchenlei/HandRefinerPortable/raw/master/mesh_graphormer/modeling/data/MANO_LEFT.pkl
    ```

3.  **Compatibility Shims (NumPy & Pillow)**:
    We modified `labs/glove2robot/postprocess/extract_hamer.py` to add shims for:
    *   **NumPy 2.x Compatibility:** Redirects `numpy._core` imports to `numpy.core` to support packages built for NumPy 2.0 running on 1.26.4.
    *   **Pillow 10+ Compatibility:** Maps deprecated `PIL.Image.LINEAR` to `BILINEAR` and `ANTIALIAS` to `LANCZOS` to fix `detectron2` crashes.

4.  **Path Corrections**:
    *   Replaced hardcoded paths (e.g., `/home/gumdev`) with `os.path.expanduser("~")` or relative paths.
    *   Updated `labs/glove2robot/config/config_extract_hamer.yaml` to point `hamer_repo_path` to the local project directory.

### Verification
You can verify the environment with:
```bash
/home/yxma/miniconda3/envs/osmo/bin/python -c "import torch; import flash_attn; import detectron2; import sam2; import hamer; import mmpose; import glove2robot; import mmint_foundationstereo; print('Environment verified!')"
```

## Special Issues

### Headless Qt Fix (General Debugging/SSH)
To prevent `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` errors in headless environments (like VS Code debugger or SSH), we added `os.environ["QT_QPA_PLATFORM"] = "offscreen"` to the top of the processing scripts. This is a general Qt issue when no X server is available.