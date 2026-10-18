import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src" / "classification"))

from dataset import train_transform

def visualize_augmentation(category="bottle", num_samples=3):
    raw_good_dir = ROOT / "data" / "raw" / "mvtec" / category / "test" / "good"
    if not raw_good_dir.exists():
        raw_good_dir = ROOT / "data" / "raw" / "mvtec" / category / "train" / "good"
    
    if not raw_good_dir.exists():
        print(f"Directory not found for category: {category}")
        return

    images = list(raw_good_dir.glob("*.png"))
    if not images:
        print("No images found.")
        return

    output_dir = ROOT / "outputs" / "augmented" / category
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transform without normalization for saving visual preview
    visual_augment_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.8),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])

    print(f"\nGenerating augmented visual samples for category: '{category}'...")

    for i, img_path in enumerate(images[:num_samples]):
        orig_img = Image.open(img_path).convert("RGB")
        resized_orig = orig_img.resize((224, 224))
        
        # Generate 2 augmented variations per sample
        aug1 = visual_augment_transform(orig_img)
        aug2 = visual_augment_transform(orig_img)

        # Concatenate horizontally: Original | Augmented #1 | Augmented #2
        combined = Image.new("RGB", (224 * 3, 224))
        combined.paste(resized_orig, (0, 0))
        combined.paste(aug1, (224, 0))
        combined.paste(aug2, (224 * 2, 0))

        save_path = output_dir / f"augmented_sample_{i+1}.jpg"
        combined.save(save_path)
        print(f"Saved augmented preview: {save_path}")

    print(f"\nAugmented results saved in folder: {output_dir}\n")

if __name__ == "__main__":
    category_name = sys.argv[1] if len(sys.argv) > 1 else "bottle"
    visualize_augmentation(category_name)
