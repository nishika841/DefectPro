import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT / "src" / "classification"))
sys.path.append(str(ROOT / "src" / "segmentation"))

from model import get_model
from unet import UNet


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE_CLASSIFICATION = 224
IMAGE_SIZE_SEGMENTATION = 256

classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

segmentation_image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_classification_model(category):

    model_path = (
        ROOT /
        "models" /
        category /
        "classification" /
        "resnet50.pth"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Classification model not found:\n{model_path}"
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


def load_segmentation_model(category):

    model_path = (
        ROOT /
        "models" /
        category /
        "segmentation" /
        "unet.pth"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Segmentation model not found:\n{model_path}"
        )

    model = UNet()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def classify_image(model, image):

    image_tensor = classification_transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    if predicted_class == 0:
        label = "Good"
    else:
        label = "Defect"

    return label, confidence


def segment_image(model, image):

    image_tensor = segmentation_image_transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        prediction = torch.sigmoid(output)

        mask = (
            prediction > 0.5
        ).float()

    return mask


def main():

    print("========================================")
    print("       DEFECT DETECTION SYSTEM")
    print("========================================")

    print("\nAvailable categories:")

    categories = sorted([
        folder.name
        for folder in (
            ROOT / "models"
        ).iterdir()
        if folder.is_dir()
    ])

    for category in categories:
        print("-", category)

    category = input(
        "\nEnter category: "
    ).strip().lower()

    if category not in categories:
        print("\nInvalid category.")
        return

    image_path = input(
        "Enter image path: "
    ).strip().strip('"')

    image_path = Path(image_path)

    if not image_path.exists():
        print("\nImage not found.")
        return

    image = Image.open(
        image_path
    ).convert("RGB")

    print("\nLoading models...")

    classification_model = (
        load_classification_model(
            category
        )
    )

    segmentation_model = (
        load_segmentation_model(
            category
        )
    )

    print("Models loaded successfully.")

    print("\nRunning classification...")

    label, confidence = classify_image(
        classification_model,
        image
    )

    print(
        f"Prediction: {label}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print("\nRunning segmentation...")

    mask = segment_image(
        segmentation_model,
        image
    )

    defect_pixels = mask.sum().item()

    if defect_pixels > 0:
        segmentation_result = "Defect region detected"
    else:
        segmentation_result = "No defect region detected"

    print(
        f"Segmentation: {segmentation_result}"
    )

    print("\n========================================")
    print("              RESULT")
    print("========================================")
    print(f"Category: {category}")
    print(f"Classification: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Segmentation: {segmentation_result}")
    print("========================================")


if __name__ == "__main__":
    main()