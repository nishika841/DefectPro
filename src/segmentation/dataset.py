import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from pathlib import Path
import random


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None, is_train=False):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"

        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.is_train = is_train

        self.images = sorted(self.image_dir.glob("*"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.is_train:
            # Joint Spatial Transformations (Image + Mask)
            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = F.rotate(image, angle)
                mask = F.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

            # Color jitter only on image
            if random.random() > 0.5:
                jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
                image = jitter(image)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask > 0).float()

        return image, mask


image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


mask_transform = transforms.Compose([
    transforms.Resize(
        (256, 256),
        interpolation=transforms.InterpolationMode.NEAREST
    ),
    transforms.ToTensor()
])