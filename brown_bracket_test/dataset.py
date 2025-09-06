from torch.utils.data import Dataset
import cv2
import glob
import matplotlib.pyplot as plt
import random
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        mask  = cv2.imread(self.mask_paths[idx])

        if image is None or mask is None:
            raise ValueError(f"Error loading image or mask at index {idx}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        binary_mask = np.any(mask > 10, axis=-1).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)

        return image, mask.float()
