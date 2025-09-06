import segmentation_models_pytorch as smp
import torch
import os
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import get_augmented_pairs, get_combined_image_mask_pairs
from sklearn.model_selection import train_test_split

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b0",  
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).cuda()

# model = smp.Unet(
#     encoder_name="resnet18",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=1,
#     activation=None
# ).cuda()  # or .to(device)

# load weights
model.load_state_dict(torch.load("./model_chkpts/unetpp_v1.pth"))

model = model.cuda()
model.eval()
# ----------------- Utilities -----------------
def add_heading(image, text):
    color = (255, 255, 255)  
    cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return image

resize_shape = (512, 512)

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(resize_shape),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------- Validation Evaluation -----------------
# Uses the val dataset images and masks fro train_test_split
def eval_with_val_ds(model, val_imgs, val_masks, image_transform, resize_shape, num_samples=5, threshold=0.5):
    print("Starting eval...")
    model.eval()
    device = next(model.parameters()).device
    os.makedirs("results", exist_ok=True)

    for i in range(num_samples):
        img_path = val_imgs[i]
        mask_path = val_masks[i]

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, resize_shape)
        input_tensor = image_transform(image_resized).unsqueeze(0).to(device)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask, resize_shape)
        mask_np = (mask_resized > 10).astype(np.uint8) * 255

        with torch.no_grad():
            pred = model(input_tensor)
            pred = torch.sigmoid(pred)
            pred_mask = (pred > threshold).float()

        pred_mask_np = pred_mask.squeeze().cpu().numpy()
        pred_mask_img = (pred_mask_np * 255).astype(np.uint8)

        defect_area = np.sum(pred_mask_np)
        is_defective = defect_area > 50 #threshold to classify as defective

        vis_image = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
        mask_color = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        pred_color = cv2.cvtColor(pred_mask_img, cv2.COLOR_GRAY2BGR)

        overlay = vis_image.copy()
        overlay[pred_mask_img > 127] = [0, 0, 255]
        blended = cv2.addWeighted(vis_image, 0.6, overlay, 0.4, 0)

        vis_image = add_heading(vis_image, "Original")
        mask_color = add_heading(mask_color, "Ground Truth")
        pred_color = add_heading(pred_color, "Prediction")
        label = "Overlay: Defective" if is_defective else "Overlay: Good"
        blended = add_heading(blended, label)

        combined = np.hstack((vis_image, mask_color, pred_color, blended))
        out_path = os.path.join("results", f"eval_result_{i}.png")
        cv2.imwrite(out_path, combined)
        print(f"Saved: {out_path}")

# ----------------- Synthetic Data Evaluation -----------------
# Contains images generated using DALL-e with test image as reference
def eval_with_synthetic_data(model, synth_data_dir, image_transform, resize_shape, threshold=0.5):
    print("Starting eval on synthetic data...")
    model.eval()
    device = next(model.parameters()).device

    image_fnames = sorted([f for f in os.listdir(synth_data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    os.makedirs("results/synthetic", exist_ok=True)

    for i, img_name in enumerate(image_fnames):
        img_path = os.path.join(synth_data_dir, img_name)

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, resize_shape)
        input_tensor = image_transform(image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input_tensor)
            pred = torch.sigmoid(pred)
            pred_mask = (pred > threshold).float()

        pred_mask_np = pred_mask.squeeze().cpu().numpy()
        pred_mask_img = (pred_mask_np * 255).astype(np.uint8)
        defect_area = np.sum(pred_mask_np)
        is_defective = defect_area > 50

        vis_image = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
        pred_color = cv2.cvtColor(pred_mask_img, cv2.COLOR_GRAY2BGR)

        overlay = vis_image.copy()
        overlay[pred_mask_img > 127] = [0, 0, 255]
        blended = cv2.addWeighted(vis_image, 0.6, overlay, 0.4, 0)

        vis_image = add_heading(vis_image, "Original")
        pred_color = add_heading(pred_color, "Prediction")
        label = "Overlay: Defective" if is_defective else "Overlay: Good"
        blended = add_heading(blended, label)

        combined = np.hstack((vis_image, pred_color, blended))
        out_path = os.path.join("results/synthetic", f"synthetic_result_{i}.png")
        cv2.imwrite(out_path, combined)
        print(f"Saved: {out_path}")

# ----------------- Dataset Setup -----------------
aug_img_dir = "./data/bracket_brown/augmented_dataset/images"
aug_mask_dir = "./data/bracket_brown/augmented_dataset/masks"
aug_image_paths, aug_mask_paths = get_augmented_pairs(aug_img_dir, aug_mask_dir)

DATA_DIR = './data/bracket_brown'
defect_types = ['bend_and_parts_mismatch', 'parts_mismatch', 'good']
image_paths, mask_paths = get_combined_image_mask_pairs(DATA_DIR, defect_types)

combined_image_paths = image_paths + aug_image_paths
combined_mask_paths  = mask_paths + aug_mask_paths
print(f"Combined dataset size: {len(combined_image_paths)} image-mask pairs.\n\n")

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    combined_image_paths,
    combined_mask_paths,
    test_size=0.2,
    random_state=42
)

# ----------------- Run Evaluations -----------------
# if agrs.synthetic:
eval_with_val_ds(model, val_imgs, val_masks, image_transform, resize_shape, num_samples=20)
# eval_with_synthetic_data(model, "./data/bracket_brown/synthetic_data", image_transform, resize_shape)