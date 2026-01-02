import sys
# Pillow shim
import PIL.Image
if not hasattr(PIL.Image, 'LINEAR'):
    PIL.Image.LINEAR = PIL.Image.BILINEAR
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = getattr(PIL.Image, 'LANCZOS', PIL.Image.BICUBIC)

try:
    import detectron2.modeling.backbone
    import os
    print(f"File: {detectron2.modeling.backbone.__file__}")
    print(f"Dir: {os.listdir(os.path.dirname(detectron2.modeling.backbone.__file__))}")
except Exception as e:
    print(f"Error: {e}")
