import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src" / "segmentation"))

from dataset import SegmentationDataset, image_transform, mask_transform
from unet import UNet

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_segmentation_category(category):
    test_dir = PROCESSED_DIR / category / "test"
    model_path = MODEL_DIR / category / "segmentation" / "unet.pth"

    if not test_dir.exists() or not model_path.exists():
        print(f"Skipping {category}: Test dataset or model file not found.")
        return

    test_dataset = SegmentationDataset(test_dir, image_transform, mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            predictions = (predictions > 0.5).float()

            intersection = (predictions * masks).sum(dim=(1, 2, 3))
            union = (predictions + masks - predictions * masks).sum(dim=(1, 2, 3))

            iou = (intersection + 1e-7) / (union + 1e-7)
            dice = (2 * intersection + 1e-7) / (predictions.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-7)

            total_iou += iou.sum().item()
            total_dice += dice.sum().item()
            count += images.size(0)

    mean_iou = total_iou / count if count > 0 else 0.0
    mean_dice = total_dice / count if count > 0 else 0.0

    print(f"\n========================================")
    print(f"  SEGMENTATION EVALUATION: {category.upper()}")
    print(f"========================================")
    print(f"Total Test Images : {count}")
    print(f"Mean IoU          : {mean_iou:.4f}")
    print(f"Mean Dice Score   : {mean_dice:.4f}")
    print(f"========================================\n")


def main():
    print(f"Using device: {DEVICE}")
    categories = sorted([folder.name for folder in PROCESSED_DIR.iterdir() if folder.is_dir()])
    
    if not categories:
        print("No categories found in data/processed.")
        return

    for category in categories:
        evaluate_segmentation_category(category)

if __name__ == "__main__":
    main()