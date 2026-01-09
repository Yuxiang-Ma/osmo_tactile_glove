#!/usr/bin/env python3
"""Test LangSAM detection on a single frame."""
import pickle
import cv2
import numpy as np
from PIL import Image

print("Loading test frame...")
with open('data/00/processed.pkl', 'rb') as f:
    data = pickle.load(f)

# Use frame 100 (should be in the processed range and have a hand)
test_frame = data['rs_color'][100]
print(f"Frame shape: {test_frame.shape}")

# Try to initialize LangSAM
print("\nInitializing LangSAM...")
try:
    from lang_sam import LangSAM
    model = LangSAM()
    print("✅ LangSAM initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize LangSAM: {e}")
    exit(1)

# Test different text prompts
prompts = ["glove.", "hand.", "glove", "hand", "tactile glove", "right hand"]

for prompt in prompts:
    print(f"\n{'='*50}")
    print(f"Testing prompt: '{prompt}'")
    print('='*50)

    try:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # Run detection
        results = model.predict([pil_image], [prompt])

        if len(results) > 0 and np.asarray(results[0]["masks"]).any():
            mask = results[0]["masks"][0]
            jidx, iidx = np.where(mask == 1)

            if len(iidx) > 0 and len(jidx) > 0:
                bbox = [np.min(iidx), np.min(jidx), np.max(iidx), np.max(jidx)]
                mask_area = np.sum(mask)
                print(f"✅ DETECTED! Bbox: {bbox}, Mask area: {mask_area} pixels")

                # Save visualization
                output_img = test_frame.copy()
                cv2.rectangle(output_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                cv2.imwrite(f'data/00/detection_test_{prompt.replace(" ", "_").replace(".", "")}.png', output_img)
                print(f"Saved visualization to data/00/detection_test_{prompt.replace(' ', '_').replace('.', '')}.png")
            else:
                print(f"❌ Empty mask")
        else:
            print(f"❌ No detection")

    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*50)
print("Test complete!")
