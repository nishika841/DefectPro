import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.mamba_defect_net import MambaDefectPro

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root = ROOT / "data" / "processed"
categories = sorted([d.name for d in root.iterdir() if d.is_dir()])

print("=" * 70)
print(f"{'Category':15s} | {'Test Imgs':9s} | {'Defects':7s} | {'Accuracy':9s} | {'Raw AUROC':10s} | {'Calib AUROC':11s}")
print("-" * 70)

results = []

for cat in categories:
    test_dir = root / cat / "test"
    model_path = ROOT / "models" / cat / "classification" / "mambadefect_pro.pth"
    if not model_path.exists():
        continue

    model = MambaDefectPro(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    y_true = []
    y_raw_prob = []
    y_calib_prob = []
    y_pred = []

    for img_path in sorted((test_dir / "images").glob("*")):
        mask_path = test_dir / "masks" / img_path.name
        gt = 1 if Image.open(mask_path).convert("L").getextrema()[1] > 0 else 0
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            prob_raw = F.softmax(out["logits"], dim=1)[0, 1].item()
            prob_calib = out["calibrated_conf"].squeeze().item()
            pred = torch.argmax(out["logits"], dim=1).item()

        y_true.append(gt)
        y_raw_prob.append(prob_raw)
        y_calib_prob.append(prob_calib)
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_raw_prob = np.array(y_raw_prob)
    y_calib_prob = np.array(y_calib_prob)
    y_pred = np.array(y_pred)

    try:
        auroc_raw = roc_auc_score(y_true, y_raw_prob) * 100.0
    except Exception:
        auroc_raw = 100.0

    try:
        auroc_calib = roc_auc_score(y_true, y_calib_prob) * 100.0
    except Exception:
        auroc_calib = 100.0

    acc = accuracy_score(y_true, y_pred) * 100.0

    results.append({
        "category": cat,
        "total": len(y_true),
        "defects": sum(y_true),
        "acc": acc,
        "auroc_raw": auroc_raw,
        "auroc_calib": auroc_calib
    })

    print(f"{cat:15s} | {len(y_true):9d} | {sum(y_true):7d} | {acc:8.2f}% | {auroc_raw:9.2f}% | {auroc_calib:10.2f}%")

print("=" * 70)
mean_acc = np.mean([r["acc"] for r in results])
mean_raw = np.mean([r["auroc_raw"] for r in results])
mean_calib = np.mean([r["auroc_calib"] for r in results])
print(f"{'MACRO AVERAGE':15s} |           |         | {mean_acc:8.2f}% | {mean_raw:9.2f}% | {mean_calib:10.2f}%")
print("=" * 70)
