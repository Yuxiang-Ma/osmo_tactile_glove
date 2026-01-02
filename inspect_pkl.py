import pickle
import numpy as np
import sys

# NumPy compatibility fix for older versions (NumPy 1.x running code expecting 2.x)
try:
    import numpy._core
except ImportError:
    try:
        import numpy.core as _core
        sys.modules['numpy._core'] = _core
        if hasattr(_core, 'numeric'):
            sys.modules['numpy._core.numeric'] = _core.numeric
    except ImportError:
        pass

try:
    path = "data/00/synced_mags_aligned.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Type of data: {type(data)}")
    if isinstance(data, list):
        print(f"Length of list: {len(data)}")
        if len(data) > 0:
            print(f"First element content: {data[0]}")
    
    arr = np.asarray(data, dtype="object")
    print(f"Array Shape: {arr.shape}")
    print(f"Array Size: {arr.size}")
    if arr.size > 0:
        print(f"First element: {arr.flatten()[0]}")

except Exception as e:
    print(f"Error: {e}")
