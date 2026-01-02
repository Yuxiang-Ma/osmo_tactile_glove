#!/bin/bash
# setup_osmo_env.sh
# Automates the setup of the 'osmo' conda environment for the Osmo Tactile Glove project.

set -e  # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
ENV_NAME="osmo"
PYTHON_VER="3.10"

# Attempt to locate conda
if [ -z "$CONDA_EXE" ]; then
    # Try common locations or assume it's in path
    CONDA_EXE=$(which conda || echo "")
fi

if [ -z "$CONDA_EXE" ]; then
    echo -e "\033[0;31mError: 'conda' not found. Please ensure Conda is installed and in your PATH.\033[0m"
    exit 1
fi

# Find the base directory of conda to source the shell script
# usually .../bin/conda -> .../etc/profile.d/conda.sh
CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

if [ ! -f "$CONDA_SH" ]; then
     echo -e "\033[0;33mWarning: Could not find conda.sh at $CONDA_SH. Trying to proceed with just 'conda' command.\033[0m"
else
     source "$CONDA_SH"
fi

# Determine PROJECT_ROOT relative to this script (assumed to be in <root>/setup/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SITE_PACKAGES_PATH="" 

# Move to project root to ensure clones and checks are in the right place
cd "$PROJECT_ROOT"
log "Working in project root: $PROJECT_ROOT"

# --- Colors ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[SETUP] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# --- 1. Create Conda Environment ---
log "Creating conda environment '$ENV_NAME' with Python $PYTHON_VER..."

# Verify conda is working
if ! command -v conda &> /dev/null; then
    error "conda command could not be found."
    exit 1
fi

if conda info --envs | grep -q "^$ENV_NAME "; then
    warn "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove it and start fresh? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Removing existing environment..."
        conda env remove -n "$ENV_NAME"
        conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
    else
        log "Using existing environment."
    fi
else
    conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
fi

log "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Determine site-packages path
SITE_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
log "Site-packages path: $SITE_PACKAGES_PATH"

# --- 2. Install PyTorch ---
log "Installing PyTorch 2.4.1 (CUDA 12.1)..."
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# --- 3. Install Flash Attention ---
log "Installing flash-attn..."
pip install flash-attn==2.7.4.post1

# --- 4. Install Detectron2 ---
log "Installing detectron2..."
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# --- 5. Install HaMeR ---
if [ ! -d "hamer" ]; then
    log "Cloning HaMeR..."
    git clone https://github.com/geopavlakos/hamer.git
else
    warn "'hamer' directory already exists. Skipping clone."
fi
log "Installing HaMeR..."
cd hamer
pip install -e .
cd ..

# --- 6. Install SAM 2 ---
if [ ! -d "sam2" ]; then
    log "Cloning SAM 2..."
    git clone https://github.com/facebookresearch/sam2.git
else
    warn "'sam2' directory already exists. Skipping clone."
fi
log "Installing SAM 2..."
cd sam2
pip install -e .
cd ..

# --- 7. Install MMPose & Open3D ---
log "Installing MMPose and Open3D..."
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpose==0.24.0"
pip install open3d

# --- 8. Fix NumPy Version ---
log "Downgrading NumPy to <2.0 to avoid binary incompatibility..."
pip install "numpy<2.0"

# --- 9. Project Path Setup (.pth) ---
log "Configuring project paths..."
PTH_FILE="$SITE_PACKAGES_PATH/osmo_local.pth"
echo "$PROJECT_ROOT" > "$PTH_FILE"
echo "$PROJECT_ROOT/labs" >> "$PTH_FILE"
log "Created $PTH_FILE"

# --- 10. FoundationStereo Symlink ---
log "Setting up FoundationStereo..."
FS_TARGET="$PROJECT_ROOT/FoundationStereo"
FS_LINK="$SITE_PACKAGES_PATH/mmint_foundationstereo"

if [ -d "$FS_TARGET" ]; then
    if [ -L "$FS_LINK" ] || [ -e "$FS_LINK" ]; then
        warn "Link $FS_LINK already exists."
    else
        ln -s "$FS_TARGET" "$FS_LINK"
        log "Symlinked $FS_TARGET to $FS_LINK"
    fi
else
    warn "Directory '$FS_TARGET' not found!"
    warn "Please clone FoundationStereo into the project root manually, then run:"
    warn "ln -s $FS_TARGET $FS_LINK"
fi

# --- 11. Final Verification ---
log "Verifying installation..."
python -c "
import sys, torch, numpy
print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}')
print(f'NumPy: {numpy.__version__}')
try:
    import flash_attn, detectron2, sam2, hamer, mmpose, glove2robot
    print('SUCCESS: All core modules imported.')
except ImportError as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

log "Setup Complete! To use the environment:"
log "conda activate $ENV_NAME"
