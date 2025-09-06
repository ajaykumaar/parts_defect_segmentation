import os
import cv2
import glob
import matplotlib.pyplot as plt
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch

from dataset import SegmentationDataset
from utils import get_augmented_pairs, get_combined_image_mask_pairs, DiceBCELoss, iou_score

image_size = 512

train_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

aug_img_dir = "./data/bracket_white/augmented_dataset/images"
aug_mask_dir = "./data/bracket_white/augmented_dataset/masks"

aug_image_paths, aug_mask_paths = get_augmented_pairs(aug_img_dir, aug_mask_dir)

DATA_DIR = './data/bracket_white'
defect_types = ['defective_painting', 'scratches', 'good']
image_paths, mask_paths = get_combined_image_mask_pairs(DATA_DIR, defect_types)

# combine aug + original
combined_image_paths = image_paths + aug_image_paths
combined_mask_paths  = mask_paths + aug_mask_paths
print(f"Combined dataset size: {len(combined_image_paths)} image-mask pairs.\n\n")


#---------------------------------------------------------------------------------------------
# Data prep and preprocessing
#---------------------------------------------------------------------------------------------

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    combined_image_paths,
    combined_mask_paths,
    test_size=0.2,
    random_state=42
)

train_dataset = SegmentationDataset(train_imgs, train_masks, transform=train_transform)
val_dataset   = SegmentationDataset(val_imgs, val_masks, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)

train_img_batch, train_mask_batch = next(iter(train_loader))
print("--"*50)

print(f"Image batch shape: {train_img_batch.shape}")
print(f"Mask batch shape: {train_mask_batch.shape} \n" )

#---------------------------------------------------------------------------------------------
# Model and Training
#---------------------------------------------------------------------------------------------
model_upp = smp.UnetPlusPlus(
    encoder_name="efficientnet-b0",  
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).cuda()

# model_u = smp.Unet(
#     encoder_name="resnet18",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=1,
#     activation=None
# ).cuda()  

##### FOR GPU CUDA #####
model = model_upp

loss_fn = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

NUM_EPOCHS = 2

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.cuda(), masks.cuda()
        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss, val_iou = 0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.cuda(), masks.cuda()
            preds = model(imgs)
            val_loss += loss_fn(preds, masks).item()
            val_iou  += iou_score(preds, masks).item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val IoU: {val_iou/len(val_loader):.4f}")


print("\nTraining complete. Saving model...")
torch.save(model.state_dict(), "./model_chkpts/unetpp_v1.pth")



