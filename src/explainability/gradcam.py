# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import cv2
# from pathlib import Path
# import sys

# ROOT = Path(__file__).resolve().parents[2]

# sys.path.append(
#     str(ROOT / "src" / "classification")
# )

# from model import get_model


# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])


# def load_model(category):

#     model_path = (
#         ROOT /
#         "models" /
#         category /
#         "classification" /
#         "resnet50.pth"
#     )

#     if not model_path.exists():
#         raise FileNotFoundError(
#             f"Model not found:\n{model_path}"
#         )

#     model = get_model()

#     model.load_state_dict(
#         torch.load(
#             model_path,
#             map_location=DEVICE
#         )
#     )

#     model = model.to(DEVICE)
#     model.eval()

#     return model


# def generate_gradcam(model, image):

#     activations = None
#     gradients = None

#     def forward_hook(module, input, output):
#         nonlocal activations
#         activations = output

#         output.register_hook(
#             backward_hook
#         )

#     def backward_hook(grad):
#         nonlocal gradients
#         gradients = grad

#     target_layer = model.layer4[-1]

#     target_layer.register_forward_hook(
#         forward_hook
#     )

#     input_tensor = transform(
#         image
#     ).unsqueeze(0).to(DEVICE)

#     output = model(input_tensor)

#     predicted_class = torch.argmax(
#         output,
#         dim=1
#     ).item()

#     probability = torch.softmax(
#         output,
#         dim=1
#     )[0, predicted_class].item()

#     model.zero_grad()

#     score = output[
#         0,
#         predicted_class
#     ]

#     score.backward()

#     if activations is None or gradients is None:
#         raise RuntimeError(
#             "Gradients capture nahi ho sake."
#         )

#     weights = gradients.mean(
#         dim=(2, 3),
#         keepdim=True
#     )

#     cam = (
#         weights * activations
#     ).sum(dim=1)

#     cam = F.relu(cam)

#     cam = cam.squeeze().detach().cpu().numpy()

#     cam = cv2.resize(
#         cam,
#         (224, 224)
#     )

#     cam -= cam.min()

#     cam /= (
#         cam.max() + 1e-8
#     )

#     return cam, predicted_class, probability


# def main():

#     print("========================================")
#     print("             GRAD-CAM")
#     print("========================================")

#     categories = sorted([
#         folder.name
#         for folder in (
#             ROOT / "models"
#         ).iterdir()
#         if folder.is_dir()
#     ])

#     print("\nAvailable categories:")

#     for category in categories:
#         print("-", category)

#     category = input(
#         "\nEnter category: "
#     ).strip().lower()

#     if category not in categories:
#         print("\nInvalid category.")
#         return

#     image_path = input(
#         "Enter image path: "
#     ).strip().strip('"')

#     image_path = Path(image_path)

#     if not image_path.exists():
#         print("\nImage not found.")
#         return

#     image = Image.open(
#         image_path
#     ).convert("RGB")

#     print("\nLoading model...")

#     model = load_model(
#         category
#     )

#     print("Model loaded successfully.")

#     print("\nGenerating Grad-CAM...")

#     cam, predicted_class, probability = (
#         generate_gradcam(
#             model,
#             image
#         )
#     )

#     if predicted_class == 0:
#         label = "Good"
#     else:
#         label = "Defect"

#     heatmap = np.uint8(
#         255 * cam
#     )

#     heatmap = cv2.applyColorMap(
#         heatmap,
#         cv2.COLORMAP_JET
#     )

#     original = cv2.imread(
#         str(image_path)
#     )

#     original = cv2.resize(
#         original,
#         (224, 224)
#     )

#     overlay = cv2.addWeighted(
#         original,
#         0.6,
#         heatmap,
#         0.4,
#         0
#     )

#     output_dir = (
#         ROOT /
#         "outputs" /
#         "gradcam" /
#         category
#     )

#     output_dir.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     output_path = (
#         output_dir /
#         "gradcam_result.jpg"
#     )

#     cv2.imwrite(
#         str(output_path),
#         overlay
#     )

#     print("\n========================================")
#     print("              RESULT")
#     print("========================================")
#     print(f"Category: {category}")
#     print(f"Prediction: {label}")
#     print(
#         f"Confidence: {probability * 100:.2f}%"
#     )
#     print("\nGrad-CAM saved at:")
#     print(output_path)
#     print("========================================")


# if __name__ == "__main__":
#     main()

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT / "src" / "classification"))

from model import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(category):
    model_path = (
        ROOT /
        "models" /
        category /
        "classification" /
        "resnet50.pth"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    model = get_model()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def generate_gradcam(model, image):
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

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_()

    output = model(input_tensor)

    predicted_class = torch.argmax(output, dim=1).item()
    probability = torch.softmax(output, dim=1)[0, predicted_class].item()

    model.zero_grad()

    score = output[0, predicted_class]
    score.backward()

    handle.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Gradients capture nahi ho sake.")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1)
    cam = F.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))

    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    return cam, predicted_class, probability


def main():
    print("========================================")
    print("             GRAD-CAM")
    print("========================================")

    excluded_folders = {
        "checkpoints",
        "classification",
        "segmentation",
        "outputs"
    }

    categories = sorted([
        folder.name
        for folder in (ROOT / "models").iterdir()
        if folder.is_dir() and folder.name not in excluded_folders
    ])

    print("\nAvailable categories:")

    for category in categories:
        print("-", category)

    category = input("\nEnter category: ").strip().lower()

    if category not in categories:
        print("\nInvalid category.")
        return

    image_path = input("Enter image path: ").strip().strip('"')
    image_path = Path(image_path)

    if not image_path.exists():
        print("\nImage not found.")
        return

    image = Image.open(image_path).convert("RGB")

    print("\nLoading model...")

    model = load_model(category)

    print("Model loaded successfully.")

    print("\nGenerating Grad-CAM...")

    cam, predicted_class, probability = generate_gradcam(
        model,
        image
    )

    label = "Good" if predicted_class == 0 else "Defect"

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    original = cv2.imread(str(image_path))
    original = cv2.resize(original, (224, 224))

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    output_dir = ROOT / "outputs" / "gradcam" / category
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / f"{image_path.stem}_gradcam.jpg"

    cv2.imwrite(
        str(output_path),
        overlay
    )

    print("\n========================================")
    print("              RESULT")
    print("========================================")
    print(f"Category: {category}")
    print(f"Prediction: {label}")
    print(f"Confidence: {probability * 100:.2f}%")

    print("\nGrad-CAM saved at:")
    print(output_path)

    print("========================================")


if __name__ == "__main__":
    main()