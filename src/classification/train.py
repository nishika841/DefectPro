# import torch
# from torch.utils.data import DataLoader
# from torch import nn, optim
# from pathlib import Path

# from dataset import DefectDataset, train_transform, val_transform
# from model import get_model


# ROOT = Path(__file__).resolve().parents[2]

# TRAIN_DIR = ROOT / "data" / "processed" / "train"
# VAL_DIR = ROOT / "data" / "processed" / "val"
# MODEL_DIR = ROOT / "models" / "classification"

# BATCH_SIZE = 16
# EPOCHS = 10
# LEARNING_RATE = 0.001

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("Device:", DEVICE)


# train_dataset = DefectDataset(
#     TRAIN_DIR,
#     transform=train_transform
# )

# val_dataset = DefectDataset(
#     VAL_DIR,
#     transform=val_transform
# )

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=0
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=0
# )


# model = get_model()
# model = model.to(DEVICE)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(
#     model.fc.parameters(),
#     lr=LEARNING_RATE
# )


# for epoch in range(EPOCHS):

#     model.train()

#     train_loss = 0
#     train_correct = 0
#     train_total = 0

#     for images, labels in train_loader:

#         images = images.to(DEVICE)
#         labels = labels.to(DEVICE)

#         optimizer.zero_grad()

#         outputs = model(images)

#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()

#         _, predicted = torch.max(outputs, 1)

#         train_total += labels.size(0)
#         train_correct += (predicted == labels).sum().item()

#     train_accuracy = 100 * train_correct / train_total

#     model.eval()

#     val_loss = 0
#     val_correct = 0
#     val_total = 0

#     with torch.no_grad():

#         for images, labels in val_loader:

#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)

#             outputs = model(images)

#             loss = criterion(outputs, labels)

#             val_loss += loss.item()

#             _, predicted = torch.max(outputs, 1)

#             val_total += labels.size(0)
#             val_correct += (predicted == labels).sum().item()

#     val_accuracy = 100 * val_correct / val_total

#     print(
#         f"Epoch [{epoch + 1}/{EPOCHS}] "
#         f"Train Loss: {train_loss / len(train_loader):.4f} "
#         f"Train Acc: {train_accuracy:.2f}% "
#         f"Val Loss: {val_loss / len(val_loader):.4f} "
#         f"Val Acc: {val_accuracy:.2f}%"
#     )


# MODEL_DIR.mkdir(parents=True, exist_ok=True)

# torch.save(
#     model.state_dict(),
#     MODEL_DIR / "resnet50_bottle.pth"
# )

# print("\nModel saved successfully.")






import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from pathlib import Path
from dataset import DefectDataset, train_transform, val_transform
from model import get_model

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

BATCH_SIZE = 16
EPOCHS = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

categories = sorted([
    folder.name
    for folder in PROCESSED_DIR.iterdir()
    if folder.is_dir()
])

print("\nCategories found:")
for category in categories:
    print("-", category)

for category in categories:

    print("\n========================================")
    print(f"Training fine-tuned classification model: {category}")
    print("========================================")

    TRAIN_DIR = PROCESSED_DIR / category / "train"
    VAL_DIR = PROCESSED_DIR / category / "val"

    train_dataset = DefectDataset(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = DefectDataset(
        VAL_DIR,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = get_model(unfreeze_layer4=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        scheduler.step()
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0.0

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0.0

        if val_accuracy >= best_val_acc:
            best_val_acc = val_accuracy
            best_model_weights = model.state_dict().copy()

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss / len(train_loader):.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Val Loss: {val_loss / len(val_loader):.4f} "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    category_model_dir = MODEL_DIR / category / "classification"
    category_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = category_model_dir / "resnet50.pth"

    if best_model_weights is not None:
        torch.save(best_model_weights, model_path)
    else:
        torch.save(model.state_dict(), model_path)

    print(f"\n{category} best model saved at: {model_path} (Best Val Acc: {best_val_acc:.2f}%)")

print("\n========================================")
print("ALL CATEGORY MODELS TRAINED & FINE-TUNED SUCCESSFULLY")
print("========================================")