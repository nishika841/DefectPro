"""
Training Pipeline for MambaDefect-Pro
-------------------------------------
Industrial Defect Detection under a ZERO-IMAGE-AUGMENTATION regime.
Integrates State-Space Modeling (SS2D), Micro-Defect Attention (BAM),
Dynamic Feature Fusion (DGDFF), and Uncertainty Quantification (UAH).
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.classification.dataset import DefectDataset, train_transform, val_transform
from src.models.mamba_defect_net import MambaDefectPro

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
LOG_FILE = ROOT / "training.log"


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(LOG_FILE)


def compute_uncertainty_loss(outputs, labels, alpha=1.0):
    """
    Composite loss function incorporating classification error and uncertainty penalty.
    Formulation inspired by PharmaNet Deep (2025) and M2U-InspectNet (2025).
    """
    logits = outputs["logits"]
    uncertainty = outputs["uncertainty"].squeeze(-1)
    
    # Standard cross entropy
    ce_loss = nn.CrossEntropyLoss()(logits, labels)
    
    # Calculate predicted error
    probs = torch.softmax(logits, dim=-1)
    pred_class = torch.argmax(probs, dim=-1)
    is_incorrect = (pred_class != labels).float()
    
    # For incorrect predictions, model should exhibit high uncertainty
    # For correct predictions, model should exhibit low uncertainty
    unc_reg = torch.mean((uncertainty - is_incorrect) ** 2)
    
    total_loss = ce_loss + 0.1 * unc_reg
    return total_loss, ce_loss, unc_reg


def train_category(category, epochs=10, batch_size=8, lr=3e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 50)
    print(f"Training MambaDefect-Pro (Zero-Augmentation) on: {category}")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 50)
    
    train_dir = PROCESSED_DIR / category / "train"
    val_dir = PROCESSED_DIR / category / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"Error: {category} processed directories not found at {train_dir}")
        return None
    
    train_dataset = DefectDataset(train_dir, transform=train_transform)
    val_dataset = DefectDataset(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Instantiate MambaDefect-Pro
    # Use pretrained=False if downloading offline, or True if connected
    try:
        model = MambaDefectPro(num_classes=2, pretrained=True).to(device)
    except Exception as e:
        print(f"Note: Pretrained weights failed ({e}), initializing standard architecture.")
        model = MambaDefectPro(num_classes=2, pretrained=False).to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        num_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss, ce, unc = compute_uncertainty_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs["logits"], dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            # Continuous live progress stream on every 2 batches
            if (batch_idx + 1) % 2 == 0 or (batch_idx + 1) == num_batches:
                current_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
                print(
                    f"  -> [Epoch {epoch+1:02d}/{epochs:02d} | Batch {batch_idx+1:02d}/{num_batches:02d}] "
                    f"Loss: {loss.item():.4f} | Running Acc: {current_acc:5.1f}%",
                    flush=True
                )
            
        scheduler.step()
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        avg_train_loss = train_loss / train_total if train_total > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        uncertainties = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss, _, _ = compute_uncertainty_loss(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs["logits"], dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                unc_batch = outputs["uncertainty"].cpu().view(-1).tolist()
                uncertainties.extend(unc_batch)
                
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        avg_val_loss = val_loss / val_total if val_total > 0 else 0.0
        mean_unc = sum(uncertainties) / len(uncertainties) if len(uncertainties) > 0 else 0.0
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            
        print(
            f"Epoch [{epoch+1:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Mean Uncertainty: {mean_unc:.4f}",
            flush=True
        )
        
    save_dir = MODEL_DIR / category / "classification"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "mambadefect_pro.pth"
    
    if best_weights is not None:
        torch.save(best_weights, save_path)
    else:
        torch.save(model.state_dict(), save_path)
        
    print(f"\n[DONE] Model saved to {save_path} with Best Val Acc: {best_val_acc:.2f}%\n")
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MambaDefect-Pro on MVTec AD")
    parser.add_argument("--category", type=str, default="zipper", help="Category name (e.g. zipper, bottle, pill)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--all", action="store_true", help="Train on all 15 categories")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip already trained categories")
    parser.add_argument("--overwrite", action="store_true", help="Force retrain existing checkpoints")
    args = parser.parse_args()
    
    if args.all:
        cats = sorted([f.name for f in PROCESSED_DIR.iterdir() if f.is_dir()])
        summary = {}
        for cat in cats:
            save_path = MODEL_DIR / cat / "classification" / "mambadefect_pro.pth"
            if save_path.exists() and args.skip_existing and not args.overwrite:
                print(f"[SKIP] Category '{cat}' checkpoint already exists ({save_path.name}). Skipping to next.", flush=True)
                continue
            acc = train_category(cat, epochs=args.epochs, batch_size=args.batch_size)
            if acc is not None:
                summary[cat] = acc
        print("\n" + "=" * 55, flush=True)
        print("MAMBADEFECT-PRO FULL BENCHMARK TRAINING SUMMARY", flush=True)
        print("=" * 55, flush=True)
        for cat, acc in summary.items():
            print(f"{cat:15s}: Best Val Acc = {acc:6.2f}%", flush=True)
        if summary:
            mean_acc = sum(summary.values()) / len(summary)
            print("-" * 55, flush=True)
            print(f"Macro Average Accuracy: {mean_acc:6.2f}%", flush=True)
        print("=" * 55, flush=True)
    else:
        train_category(args.category, epochs=args.epochs, batch_size=args.batch_size)
