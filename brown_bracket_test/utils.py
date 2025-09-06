import os
import torch
import torch.nn as nn


def get_combined_image_mask_pairs(data_dir, defect_types):
    image_paths = []
    mask_paths = []

    for defect in defect_types:
        image_dir = os.path.join(data_dir, 'test', defect)
        mask_dir  = os.path.join(data_dir, 'ground_truth', defect)

        for img_file in os.listdir(image_dir):
            if img_file.endswith((".png", ".jpg")):
                img_path = os.path.join(image_dir, img_file)
                mask_file = img_file.split('.')[0] + "_mask.png"
                mask_path = os.path.join(mask_dir, mask_file)

                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                    
        print(f"Defect: {defect}  |  num samples:  {len(os.listdir(image_dir))}")

    return image_paths, mask_paths


def get_augmented_pairs(img_dir, mask_dir):
    img_paths, mask_paths = [], []
    for img_file in os.listdir(img_dir):
        if img_file.endswith(".png"):
            base = img_file.split(".")[0]
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, f"{base}_mask.png")
            if os.path.exists(mask_path):
                img_paths.append(img_path)
                mask_paths.append(mask_path)
    return img_paths, mask_paths




# Loss function
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)

        smooth = 1e-6
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice

        return bce_loss + dice_loss
    

def iou_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds) > threshold
    targets = targets.bool()

    intersection = (preds & targets).float().sum((1, 2, 3))
    union = (preds | targets).float().sum((1, 2, 3))

    return ((intersection + 1e-6) / (union + 1e-6)).mean()
