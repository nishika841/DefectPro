import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

sys.path.append(str(ROOT / "src" / "classification"))
sys.path.append(str(ROOT / "src" / "segmentation"))

from src.models.mamba_defect_net import MambaDefectPro
from model import get_model
from unet import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE_CLASSIFICATION = 224
IMAGE_SIZE_SEGMENTATION = 256

# Industrial inspection thresholds (calibrated to counter training class imbalance)
DEFECT_PROB_THRESHOLD = 0.35      # 35% defect probability indicates significant defect presence
EPISTEMIC_UNC_THRESHOLD = 0.12    # Uncertainty >= 0.12 indicates feature anomaly (normal goods <= 0.005)
SEGMENTATION_PIXEL_THRESHOLD = 50 # Minimum pixel count to confirm localized defect
SEGMENTATION_CONF_THRESHOLD = 0.25 # Calibrated U-Net sensitivity threshold

classification_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE_CLASSIFICATION, IMAGE_SIZE_CLASSIFICATION)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

segmentation_image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE_SEGMENTATION, IMAGE_SIZE_SEGMENTATION)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_classification_model(category):
    """
    Loads MambaDefect-Pro as the primary SOTA model.
    Falls back to baseline ResNet50 if Mamba checkpoint is not available.
    """
    mamba_path = ROOT / "models" / category / "classification" / "mambadefect_pro.pth"
    resnet_path = ROOT / "models" / category / "classification" / "resnet50.pth"

    if mamba_path.exists():
        model = MambaDefectPro(num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(mamba_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        return model, "MambaDefect-Pro"
    elif resnet_path.exists():
        model = get_model()
        model.load_state_dict(torch.load(resnet_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        return model, "ResNet50 (Baseline)"
    else:
        raise FileNotFoundError(
            f"No classification checkpoint found for '{category}' in {mamba_path.parent}"
        )


def load_segmentation_model(category):
    """Loads U-Net segmentation model for pixel-accurate defect localization."""
    model_path = ROOT / "models" / category / "segmentation" / "unet.pth"
    if not model_path.exists():
        return None

    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def classify_image(model, model_type, image):
    """
    Runs classification inference.
    Returns defect probability, good probability, and epistemic uncertainty.
    """
    image_tensor = classification_transform(image).unsqueeze(0).to(DEVICE)

    if model_type == "MambaDefect-Pro":
        with torch.no_grad():
            outputs = model(image_tensor)
            logits = outputs["logits"]
            probabilities = F.softmax(logits, dim=1)
            good_prob = probabilities[0, 0].item()
            defect_prob = probabilities[0, 1].item()
            uncertainty = outputs["uncertainty"].squeeze().item()
            calibrated_conf = outputs["calibrated_conf"].squeeze().item()

        return good_prob, defect_prob, uncertainty, calibrated_conf
    else:
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            good_prob = probabilities[0, 0].item()
            defect_prob = probabilities[0, 1].item()

        return good_prob, defect_prob, 0.0, defect_prob


def segment_image(model, image):
    """
    Runs U-Net segmentation for pixel-level defect mask using calibrated threshold.
    """
    if model is None:
        return torch.zeros((1, 1, IMAGE_SIZE_SEGMENTATION, IMAGE_SIZE_SEGMENTATION), device=DEVICE)

    image_tensor = segmentation_image_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output)
        # Use calibrated 0.25 threshold to reliably detect subtle/thin defects
        mask = (prediction > SEGMENTATION_CONF_THRESHOLD).float()
    return mask


def fuse_decisions(good_prob, defect_prob, uncertainty, defect_pixels, model_type):
    """
    Multi-Modal Decision Fusion (PharmaNet Deep 2025 standard).
    Combines:
    1. Defect probability from Mamba classification
    2. Epistemic uncertainty trigger (signals feature distribution shifts)
    3. Pixel-level defect localization from U-Net
    """
    reasons = []

    has_prob_signal = defect_prob >= DEFECT_PROB_THRESHOLD
    has_unc_signal = (model_type == "MambaDefect-Pro") and (uncertainty >= EPISTEMIC_UNC_THRESHOLD)
    has_seg_signal = defect_pixels >= SEGMENTATION_PIXEL_THRESHOLD

    if has_prob_signal:
        reasons.append(f"Classification Score ({defect_prob*100:.1f}%)")
    if has_unc_signal:
        reasons.append(f"High Epistemic Uncertainty ({uncertainty:.4f})")
    if has_seg_signal:
        reasons.append(f"Localized Defect Mask ({defect_pixels:.0f} px)")

    is_defect = has_prob_signal or has_unc_signal or has_seg_signal

    if is_defect:
        label = "Defect"
        # Confidence reflects the most decisive indicator
        candidates = [defect_prob]
        if has_unc_signal:
            candidates.append(min(0.99, 0.50 + uncertainty))
        if has_seg_signal:
            candidates.append(min(0.99, 0.60 + (defect_pixels / 1000.0) * 0.35))
        confidence = max(candidates)
        decision_details = " + ".join(reasons)
    else:
        label = "Good"
        confidence = good_prob
        decision_details = "Normal Texture Verified (No Anomaly Triggers)"

    return label, confidence, decision_details


def generate_gradcam(model, model_type, image):
    """Generates Grad-CAM explainability heatmap."""
    activations = None
    gradients = None

    def extract_grad(grad):
        nonlocal gradients
        gradients = grad

    def forward_hook(module, input, output):
        nonlocal activations
        output.requires_grad_()
        activations = output
        output.register_hook(extract_grad)

    target_layer = model.layer4[-1]
    handle = target_layer.register_forward_hook(forward_hook)

    input_tensor = classification_transform(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_()

    if model_type == "MambaDefect-Pro":
        output_dict = model(input_tensor)
        logits = output_dict["logits"]
    else:
        logits = model(input_tensor)

    # Compute gradients with respect to defect class (index 1) to highlight defect regions
    score = logits[0, 1]

    model.zero_grad()
    score.backward()

    handle.remove()

    if activations is None or gradients is None:
        return np.zeros((IMAGE_SIZE_CLASSIFICATION, IMAGE_SIZE_CLASSIFICATION))

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1)
    cam = F.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = cv2.resize(cam, (IMAGE_SIZE_CLASSIFICATION, IMAGE_SIZE_CLASSIFICATION))

    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    return cam


def save_unified_visualization(orig_image, cam, mask, label, confidence, uncertainty,
                              defect_pixels, category, model_type, decision_details, output_path):
    orig_np = np.array(orig_image.convert("RGB").resize((224, 224)))

    # 1. Grad-CAM Overlay
    heatmap = np.uint8(255 * cam)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    gradcam_overlay = cv2.addWeighted(orig_np, 0.6, heatmap_colored, 0.4, 0)

    # 2. Segmentation Mask Overlay
    mask_np = mask.squeeze().detach().cpu().numpy()
    mask_resized = cv2.resize(mask_np, (224, 224), interpolation=cv2.INTER_NEAREST)

    seg_overlay = orig_np.copy()
    if defect_pixels > 0:
        red_mask = np.zeros_like(orig_np)
        red_mask[:, :] = [255, 30, 30]
        defect_idx = mask_resized > 0.5
        seg_overlay[defect_idx] = cv2.addWeighted(orig_np[defect_idx], 0.35, red_mask[defect_idx], 0.65, 0)

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor("#0f172a")

    status_color = "#ef4444" if label == "Defect" else "#22c55e"
    unc_text = f" | Epistemic Unc: {uncertainty:.4f}" if model_type == "MambaDefect-Pro" else ""

    fig.suptitle(
        f"DefectPro Pipeline ({model_type}): {category.upper()} | {label.upper()} ({confidence*100:.2f}% Conf{unc_text})",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.98
    )

    # Panel 1: Original
    axes[0].imshow(orig_np)
    axes[0].set_title("1. Input Image", fontsize=13, fontweight="bold", color="white", pad=14)
    axes[0].axis("off")

    # Panel 2: Grad-CAM Explainability
    axes[1].imshow(gradcam_overlay)
    axes[1].set_title(f"2. Grad-CAM Attention\n[{label} ({confidence*100:.2f}%)]", 
                      fontsize=13, fontweight="bold", color=status_color, pad=14)
    axes[1].axis("off")

    # Panel 3: U-Net Segmentation
    axes[2].imshow(seg_overlay)
    seg_title = f"Defect Area: {defect_pixels:.0f} px" if defect_pixels > 0 else "No Defect Detected"
    axes[2].set_title(f"3. U-Net Segmentation\n[{seg_title}]", 
                      fontsize=13, fontweight="bold", color=status_color if defect_pixels > 0 else "#22c55e", pad=14)
    axes[2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    except OSError:
        fallback = output_path.parent / f"{output_path.stem}_{int(time.time())}{output_path.suffix}"
        plt.savefig(fallback, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        print(f" [NOTICE] Could not overwrite {output_path.name} (file open in viewer). Saved to {fallback.name} instead.")
    finally:
        plt.close(fig)


def run_pipeline(category, image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"\n[ERROR] Image not found at: {image_path}")
        return

    print("\n" + "=" * 60)
    print(f"       DEFECTPRO UNIFIED INFERENCE PIPELINE")
    print("=" * 60)
    print(f" Category : {category}")
    print(f" Image    : {image_path.name}")
    print(f" Device   : {DEVICE}")
    print("-" * 60)

    image = Image.open(image_path).convert("RGB")

    print("[1/3] Loading Model Weights...")
    classification_model, model_type = load_classification_model(category)
    segmentation_model = load_segmentation_model(category)
    print(f"      -> Classification Engine : {model_type}")
    print(f"      -> Localization Engine   : {'U-Net (256x256)' if segmentation_model else 'None'}")

    print(f"[2/3] Running {model_type} Classification & Grad-CAM...")
    good_prob, defect_prob, uncertainty, calibrated_conf = classify_image(
        classification_model, model_type, image
    )
    cam = generate_gradcam(classification_model, model_type, image)

    print("[3/3] Running U-Net Defect Segmentation...")
    mask = segment_image(segmentation_model, image)
    defect_pixels = mask.sum().item()

    if defect_pixels > 0:
        segmentation_status = f"Defect Region Localized ({defect_pixels:.0f} pixels)"
    else:
        segmentation_status = "No Defect Region Detected"

    # Multi-Modal Decision Fusion
    label, confidence, decision_details = fuse_decisions(
        good_prob, defect_prob, uncertainty, defect_pixels, model_type
    )

    # Save Unified Visual Report
    output_dir = ROOT / "outputs" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    unified_output_path = output_dir / f"{category}_{image_path.stem}_unified.png"
    default_output_path = output_dir / "unified_result.png"

    save_unified_visualization(
        image, cam, mask, label, confidence, uncertainty, defect_pixels,
        category, model_type, decision_details, unified_output_path
    )

    # Safely copy to default unified_result.png without crashing if open in viewer
    try:
        shutil.copyfile(unified_output_path, default_output_path)
    except OSError:
        pass

    print("\n" + "=" * 60)
    print("              PIPELINE INFERENCE RESULTS")
    print("=" * 60)
    print(f" Category         : {category.upper()}")
    print(f" Model Engine     : {model_type}")
    print(f" Verdict          : {label.upper()}")
    print(f" Confidence       : {confidence * 100:.2f}%")
    print(f" Decision Basis   : {decision_details}")
    if model_type == "MambaDefect-Pro":
        print(f" Defect Prob.     : {defect_prob * 100:.2f}%")
        print(f" Good Prob.       : {good_prob * 100:.2f}%")
        print(f" Epistemic Unc.   : {uncertainty:.5f}")
    print(f" Explainability   : Grad-CAM attention heatmap generated")
    print(f" Segmentation     : {segmentation_status}")
    print("-" * 60)
    print(f" [OUTPUT] Visual Reports Saved At:")
    print(f"   -> {unified_output_path}")
    print(f"   -> {default_output_path}")
    print("=" * 60 + "\n")

    # Auto-open category-specific report
    try:
        import os
        os.startfile(str(unified_output_path))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Unified Industrial Defect Detection CLI")
    parser.add_argument("category", nargs="?", default=None, help="Category name (e.g. zipper, wood, bottle)")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to input image")
    parser.add_argument("--demo", action="store_true", help="Run benchmark demo on Zipper defect sample")

    args = parser.parse_args()

    if args.demo:
        demo_category = "zipper"
        demo_image = ROOT / "data" / "raw" / "mvtec" / "zipper" / "test" / "squeezed_teeth" / "010.png"
        run_pipeline(demo_category, demo_image)
        return

    category = args.category
    image_path = args.image_path

    available_categories = sorted([
        folder.name
        for folder in (ROOT / "models").iterdir()
        if folder.is_dir() and folder.name not in {"checkpoints", "classification", "segmentation", "outputs"}
    ])

    if category is not None and image_path is None:
        potential_path = Path(category)
        if potential_path.exists() and potential_path.is_file():
            path_parts = [p.lower() for p in potential_path.parts]
            detected_category = None
            for cat in available_categories:
                if cat in path_parts:
                    detected_category = cat
                    break
            if detected_category:
                image_path = str(potential_path)
                category = detected_category
                print(f"[INFO] Auto-detected category: '{category}' from file path.")

    if category is None or image_path is None:
        print("========================================")
        print("       DEFECT DETECTION SYSTEM")
        print("========================================")

        categories = available_categories

        print("\nAvailable categories:")
        for cat in categories:
            print("-", cat)

        category = input("\nEnter category: ").strip().lower()
        if category not in categories:
            print("\nInvalid category.")
            return

        image_path = input("Enter image path: ").strip().strip('"')

    run_pipeline(category, image_path)


if __name__ == "__main__":
    main()