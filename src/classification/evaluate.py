import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src" / "classification"))

from dataset import DefectDataset, val_transform
from model import get_model

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_category(category):
    test_dir = PROCESSED_DIR / category / "test"
    model_path = MODEL_DIR / category / "classification" / "resnet50.pth"

    if not test_dir.exists() or not model_path.exists():
        print(f"Skipping {category}: Test dataset or model file not found.")
        return

    test_dataset = DefectDataset(test_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Metrics computation: 0 = Good, 1 = Defect
    # Confusion Matrix:
    # TN: True Good (Actual Good, Pred Good)
    # FP: False Defect (Actual Good, Pred Defect)
    # FN: False Good (Actual Defect, Pred Good)
    # TP: True Defect (Actual Defect, Pred Defect)
    tn, fp, fn, tp = 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for p, l in zip(predicted, labels):
                p_item, l_item = p.item(), l.item()
                if l_item == 0 and p_item == 0:
                    tn += 1
                elif l_item == 0 and p_item == 1:
                    fp += 1
                elif l_item == 1 and p_item == 0:
                    fn += 1
                elif l_item == 1 and p_item == 1:
                    tp += 1

    total = tn + fp + fn + tp
    correct = tn + tp
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"\n========================================")
    print(f"  CLASSIFICATION EVALUATION: {category.upper()}")
    print(f"========================================")
    print(f"Total Test Images : {total}")
    print(f"Overall Accuracy  : {accuracy:.2f}%\n")

    print("CONFUSION MATRIX:")
    print("----------------------------------------")
    print(f"                  Predicted GOOD   Predicted DEFECT")
    print(f"Actual GOOD   :       {tn:5d}           {fp:5d}")
    print(f"Actual DEFECT :       {fn:5d}           {tp:5d}")
    print("----------------------------------------")

    print(f"Precision (Defect) : {precision:.2f}%")
    print(f"Recall (Defect)    : {recall:.2f}%")
    print(f"F1-Score (Defect)  : {f1:.2f}%")
    print("========================================\n")


def main():
    print(f"Using device: {DEVICE}")
    categories = sorted([folder.name for folder in PROCESSED_DIR.iterdir() if folder.is_dir()])
    
    if not categories:
        print("No categories found in data/processed.")
        return

    for category in categories:
        evaluate_category(category)

if __name__ == "__main__":
    main()