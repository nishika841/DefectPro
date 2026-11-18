# AnomaNet (DefectPro) — Xebia Showcase Station

A high-contrast, interactive Industrial Defect Inspection Station designed for quick evaluation by judges in under 90 seconds.

Operates **100% locally and offline** using saved, verified prediction outputs. Requires zero dataset downloads, zero PyTorch weights/checkpoints, zero GPU, and zero external web services.

---

## ⚡ Quick Launch Command

Run from the project root directory:

```bash
# Recommended one-click launcher (starts local server & auto-opens browser):
python run_showcase.py
```

*Alternative (standard Python HTTP server):*
```bash
python -m http.server 8000
# Then open: http://localhost:8000/showcase/index.html
```

*Alternative (Direct browser open):*
Double-click `showcase/index.html` in Windows Explorer (works completely offline under `file:///` protocol).

---

## ⏱️ 90-Second Judge Walkthrough

| Time | Action | What the Judge Sees & Evaluates |
| :--- | :--- | :--- |
| **00:00 – 00:20** | **1. Opening Sample: Cracked Tile** | - App starts in full-screen dark industrial style with high-contrast machine-vision UI.<br>- Hit **`INSPECT SAMPLE`** (or press `[Space]`).<br>- Watch the 1.2-second laser scanning line sweep across the image.<br>- The large **`FLAGGED`** verdict banner illuminates in vibrant hazard red, readable from across a room.<br>- Switch views: press `[1]` for Original, `[2]` for U-Net Segmentation (1,477 px localized crack), and `[3]` for Grad-CAM attention heatmap. |
| **00:20 – 00:50** | **2. Edge-Case Deep-Dive: Toothbrush** | - Click sample card 2: **Toothbrush (Hairline Bristle Defect)**.<br>- Notice the **Critical Safety Comparison Card** appears automatically:<br>&nbsp;&nbsp;&bull; *Naive 50% cutoff classifier*: **`46.4% Defect → FALSE PASS`** (defective bristle escapes factory!).<br>&nbsp;&nbsp;&bull; *AnomaNet / DefectPro Fusion*: **`FLAGGED (99.00% Conf)`**.<br>- **Explain the science**: At 46.4%, standard softmax is blind to subtle defects. But Epistemic Uncertainty ($\sigma^2$) spiked **250&times;** to `0.5255` (safety threshold: $0.12$) and U-Net caught `574 defect pixels`.<br>- Press `[Space]` to inspect and verify the **`FLAGGED`** verdict and pixel mask. |
| **00:50 – 00:70** | **3. Passing Reference Standard** | - Click sample card 3: **Toothbrush (Golden Standard)**.<br>- Press `[Space]` to inspect.<br>- The banner illuminates in glowing emerald green **`PASS (99.99% Conf)`**.<br>- Point out telemetry: Epistemic Uncertainty is flat at baseline ($\sigma^2 = 0.0021$) and U-Net defect pixels are `0 px`. |
| **00:70 – 00:90** | **4. Batch Conveyor Simulation** | - Click **`RUN BATCH`** (or press `[B]`).<br>- The bottom conveyor drawer deploys showing 7 verified factory parts moving along the simulated line.<br>- Click any item (e.g. *Grid Wire Anomaly*, *Capsule Smudge*, *Hazelnut Borer Hole*) to instantly mount and inspect it.<br>- Conclude: Operates under a strict **Zero-Image-Augmentation paradigm** with linear $O(N)$ Mamba State-Space complexity. |

---

## ⌨️ Station Keyboard Shortcuts

- `[SPACE]` : Trigger inspection scan on mounted sample
- `[R]`     : Reset station back to ready state
- `[1]`     : Switch to Original Sensor View
- `[2]`     : Switch to U-Net Segmentation Mask View
- `[3]`     : Switch to Grad-CAM Attention Heatmap View
- `[4]`     : Switch to Full 3-Panel Saved Inspection Report
- `[B]`     : Toggle Simulated Batch Conveyor Belt

---

## 🛡️ Architecture & Verification Note

All visual assets and telemetry data displayed in this station are direct crops and saved outputs from the trained models (`outputs/predictions/` and `README.md` benchmark tables). No synthetic masks are drawn or hallucinated.

---

## 📊 Live Verified Benchmark Results (All 15 MVTec AD Classes)

- **Macro Test Accuracy**: **`96.62%`**
- **Macro Image AUROC**: **`97.49%`**
- **Micro Test Accuracy**: **`96.70%`** (703 / 727 correct test predictions across all 15 industrial classes)
- **Zero-Image-Augmentation Paradigm**: Feature-Space Latent Perturbation (FSLP) preserves factory sub-pixel fidelity.

| Category | Type | Test Images | Test Accuracy | Calibrated AUROC | Performance Tier |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Grid** | Texture | 36 | **100.00%** 🌟 | **100.00%** 🌟 | Perfect Discrimination (Dual Ensemble SOTA) |
| **Screw** | Object | 64 | **100.00%** 🌟 | **100.00%** 🌟 | Perfect Discrimination (MambaDefect-Pro SOTA) |
| **Tile** | Texture | 51 | **100.00%** 🌟 | **100.00%** 🌟 | Perfect Discrimination |
| **Zipper** | Object | 57 | **100.00%** 🌟 | **100.00%** 🌟 | Perfect Discrimination |
| **Metal Nut** | Object | 39 | **100.00%** 🌟 | **100.00%** 🌟 | Perfect Discrimination (ResNet-50 Optimal) |
| **Hazelnut** | Object | 56 | 98.21% | **100.00%** 🌟 | Perfect AUROC |
| **Transistor** | Object | 68 | **97.06%** | **89.38%** | Robust SOTA Ensemble ($>97\%$) |
| **Bottle** | Object | 31 | 96.77% | **98.64%** | Near-Perfect ($>98\%$) |
| **Leather** | Texture | 52 | **96.15%** | **100.00%** 🌟 | Perfect AUROC (ResNet-50 Optimal) |
| **Carpet** | Texture | 48 | **95.83%** | **99.73%** 🌟 | High Precision Spatial U-Net ($>95\%$) |
| **Toothbrush** | Object | 17 | **94.12%** | **96.67%** | Multi-Modal Decision Fused |
| **Wood** | Texture | 32 | 93.75% | **94.33%** | High Precision |
| **Cable** | Object | 79 | **93.67%** | **93.76%** | High Precision Ensemble ($>93\%$) |
| **Capsule** | Object | 42 | 92.86% | **97.94%** | High Precision ($>97\%$) |
| **Pill** | Object | 55 | **90.91%** | **91.91%** | Dual Ensemble SOTA ($>90\%$) |
| **MACRO AVERAGE** | — | **727** | **96.62%** | **97.49%** | **SOTA Zero-Augmentation (703/727, 96.70% Micro)** |


