import sys
import os
import pickle
import numpy

# Force shim for loading
try:
    import numpy.core as _core
    sys.modules['numpy._core'] = _core
    if hasattr(_core, 'numeric'):
        sys.modules['numpy._core.numeric'] = _core.numeric
    # Patch numpy
    if not hasattr(numpy, '_core'):
        numpy._core = _core
except Exception as e:
    print(f"Shim setup failed: {e}")

def fix_file(filepath):
    print(f"Fixing {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Re-save
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

data_dir = "data/00/"
files = ["left_ir_aligned.pkl", "right_ir_aligned.pkl", "rgbs_aligned.pkl", "synced_mags_aligned.pkl"]

for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        fix_file(path)
    else:
        print(f"File not found: {path}")
