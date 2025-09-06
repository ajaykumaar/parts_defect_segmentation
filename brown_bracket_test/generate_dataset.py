import os
import cv2
import glob
import matplotlib.pyplot as plt
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import get_combined_image_mask_pairs

# Generate empty masks for good images

# Paths
GOOD_IMG_DIR = "./data/bracket_brown/test/good"
MASK_SAVE_DIR = "./data/bracket_brown/ground_truth/good"

os.makedirs(MASK_SAVE_DIR, exist_ok=True)
good_images = os.listdir(GOOD_IMG_DIR)  #[:20]
print("Generating empty masks for good images...\n")
print(f"Number of good images: {len(good_images)}")

for img_name in good_images:
    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(GOOD_IMG_DIR, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load image: {img_name}")
            continue

        height, width = img.shape[:2]

        # all zero masks
        mask = np.zeros((height, width, 3), dtype=np.uint8)

        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_mask.png"
        mask_path = os.path.join(MASK_SAVE_DIR, mask_name)

        cv2.imwrite(mask_path, mask)

        print(f"Saved: {mask_path}")
print("\nEmpty masks generated for all good images.\n")



DATA_DIR = './data/bracket_brown'
defect_types = ['bend_and_parts_mismatch', 'parts_mismatch', 'good']
image_paths, mask_paths = get_combined_image_mask_pairs(DATA_DIR, defect_types)
print("\nBefore augmentation: ")
print(f"Augmented dataset size: {len(image_paths)} image-mask pairs. \n")

#### Generate augmented data
from tqdm import tqdm

original_image_paths = image_paths  
original_mask_paths = mask_paths

aug_img_dir = "./data/bracket_brown/augmented_dataset/images"
aug_mask_dir = "./data/bracket_brown/augmented_dataset/masks"
os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

augmentations_per_image = 3
save_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.05, p=0.05),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
])

counter = 0
for i in tqdm(range(len(original_image_paths))):
    img_path = original_image_paths[i]
    mask_path = original_mask_paths[i]

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if image is None or mask is None:
        print(f"[Skipping broken] {img_path} or {mask_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    binary_mask = np.any(mask > 10, axis=-1).astype(np.uint8)

    for j in range(augmentations_per_image):
        augmented = save_transform(image=image, mask=binary_mask)
        aug_img = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
        aug_mask = (augmented["mask"] * 255).astype(np.uint8)

        img_filename = f"sample_{counter:04d}.png"
        mask_filename = f"sample_{counter:04d}_mask.png"

        cv2.imwrite(os.path.join(aug_img_dir, img_filename), aug_img)
        cv2.imwrite(os.path.join(aug_mask_dir, mask_filename), aug_mask)

        counter += 1

print("\nAfter augmentation: ")
print(f"Augmented dataset size: {counter} image-mask pairs.")
print(f"Total dataset size (original + augmented): {len(original_image_paths) + counter} image-mask pairs.\n")
print("--"*50)
print("\n")


