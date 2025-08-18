import os
import cv2
import numpy as np
from tqdm import tqdm

OUTPUT_DIR = "src/data/normal"
NUM_IMAGES = 500
IMG_SIZE = (200, 200)  

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_brushed_steel(width, height):
    """
    Generate a synthetic brushed steel texture using noise and motion blur.
    """
    # Base uniform gray surface
    base_gray = np.random.randint(160, 200)  # brightness range
    img = np.full((height, width, 3), base_gray, dtype=np.uint8)

    # Add Gaussian noise
    noise = np.random.normal(0, np.random.randint(15, 35), (height, width, 1)).astype(np.int16)
    steel = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Apply horizontal or vertical motion blur
    kernel_size = np.random.choice([9, 13, 15, 21])  # random streak length
    kernel = np.zeros((kernel_size, kernel_size))
    if np.random.rand() < 0.5:
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    else:
        kernel[:, int((kernel_size-1)/2)] = np.ones(kernel_size)
    kernel /= kernel_size
    steel = cv2.filter2D(steel, -1, kernel)

    # Random brightness/contrast shift
    alpha = np.random.uniform(0.9, 1.1)  # contrast
    beta = np.random.randint(-10, 10)    # brightness
    steel = np.clip(alpha * steel + beta, 0, 255).astype(np.uint8)

    return steel

if __name__ == "__main__":
    print(f"Generating {NUM_IMAGES} synthetic clean steel images in '{OUTPUT_DIR}'...")
    for i in range(1, NUM_IMAGES + 1):
        img = generate_brushed_steel(IMG_SIZE[0], IMG_SIZE[1])
        filename = os.path.join(OUTPUT_DIR, f"synthetic_clean_{i:03d}.jpg")
        cv2.imwrite(filename, img)
    print("✅ Done! Synthetic clean steel images generated.")
