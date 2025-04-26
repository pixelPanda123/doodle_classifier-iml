import numpy as np
import os
import cv2

# --- CONFIG ---
INPUT_DIR = "npy_files"    # where your .npy files are
OUTPUT_DIR = "data"        # where we will save images
CATEGORIES = ["apple", "banana", "basketball", "book"]
NUM_IMAGES = 5000          # number of images to extract per category

os.makedirs(OUTPUT_DIR, exist_ok=True)

for category in CATEGORIES:
    # create folder for each class
    category_path = os.path.join(OUTPUT_DIR, category)
    os.makedirs(category_path, exist_ok=True)

    npy_path = os.path.join(INPUT_DIR, f"{category}.npy")
    print(f"Loading {npy_path}...")
    
    # load numpy array
    images = np.load(npy_path)

    # limit how many images we use (optional)
    images = images[:NUM_IMAGES]

    # save each image
    for idx, img_array in enumerate(images):
        img_path = os.path.join(category_path, f"{category}_{idx}.png")
        img_array = cv2.resize(img_array, (28, 28))
        cv2.imwrite(img_path, img_array)

    print(f"Saved {len(images)} images for {category}!")

print("✅ Conversion complete!")

