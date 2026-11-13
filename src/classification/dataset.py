import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.transform = transform

        self.images = sorted(self.image_dir.glob("*"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Fast C-level max calculation (instantaneous, avoids 1M list allocations)
        label = 1 if mask.getextrema()[1] > 0 else 0

        if self.transform:
            image = self.transform(image)

        return image, label


# Zero Image Augmentation: Strict standardization pipeline as justified in ADR
# Prevents micro-defect distortion, preserves 45-degree illumination cues,
# and avoids synthetic distribution bias (Ref: PharmaNet Deep 2025, Sec 4.1).
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])