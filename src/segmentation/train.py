# # import torch
# # from torch.utils.data import DataLoader
# # from torch import nn, optim
# # from pathlib import Path

# # from dataset import SegmentationDataset, image_transform, mask_transform
# # from unet import UNet


# # ROOT = Path(__file__).resolve().parents[2]

# # TRAIN_DIR = ROOT / "data" / "processed" / "train"
# # VAL_DIR = ROOT / "data" / "processed" / "val"
# # MODEL_DIR = ROOT / "models" / "segmentation"

# # BATCH_SIZE = 8
# # EPOCHS = 20
# # LEARNING_RATE = 0.001

# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # print("Device:", DEVICE)


# # train_dataset = SegmentationDataset(
# #     TRAIN_DIR,
# #     image_transform,
# #     mask_transform
# # )

# # val_dataset = SegmentationDataset(
# #     VAL_DIR,
# #     image_transform,
# #     mask_transform
# # )


# # train_loader = DataLoader(
# #     train_dataset,
# #     batch_size=BATCH_SIZE,
# #     shuffle=True,
# #     num_workers=0
# # )

# # val_loader = DataLoader(
# #     val_dataset,
# #     batch_size=BATCH_SIZE,
# #     shuffle=False,
# #     num_workers=0
# # )


# # model = UNet().to(DEVICE)

# # criterion = nn.BCEWithLogitsLoss()

# # optimizer = optim.Adam(
# #     model.parameters(),
# #     lr=LEARNING_RATE
# # )


# # for epoch in range(EPOCHS):

# #     model.train()

# #     train_loss = 0

# #     for images, masks in train_loader:

# #         images = images.to(DEVICE)
# #         masks = masks.to(DEVICE)

# #         optimizer.zero_grad()

# #         outputs = model(images)

# #         loss = criterion(outputs, masks)

# #         loss.backward()

# #         optimizer.step()

# #         train_loss += loss.item()


# #     model.eval()

# #     val_loss = 0

# #     with torch.no_grad():

# #         for images, masks in val_loader:

# #             images = images.to(DEVICE)
# #             masks = masks.to(DEVICE)

# #             outputs = model(images)

# #             loss = criterion(outputs, masks)

# #             val_loss += loss.item()


# #     train_loss /= len(train_loader)
# #     val_loss /= len(val_loader)

# #     print(
# #         f"Epoch [{epoch + 1}/{EPOCHS}] "
# #         f"Train Loss: {train_loss:.4f} "
# #         f"Val Loss: {val_loss:.4f}"
# #     )


# # MODEL_DIR.mkdir(parents=True, exist_ok=True)

# # torch.save(
# #     model.state_dict(),
# #     MODEL_DIR / "unet_bottle.pth"
# # )

# # print("\nU-Net model saved successfully.")


# import torch
# from torch.utils.data import DataLoader
# from torch import nn, optim
# from pathlib import Path
# from dataset import SegmentationDataset, image_transform, mask_transform
# from unet import UNet

# ROOT = Path(__file__).resolve().parents[2]
# PROCESSED_DIR = ROOT / "data" / "processed"
# MODEL_DIR = ROOT / "models"

# BATCH_SIZE = 8
# EPOCHS = 20
# LEARNING_RATE = 0.001

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", DEVICE)

# categories = sorted([
#     folder.name
#     for folder in PROCESSED_DIR.iterdir()
#     if folder.is_dir()
# ])

# print("\nCategories found:")
# for category in categories:
#     print("-", category)

# for category in categories:

#     print("\n========================================")
#     print(f"Training segmentation model: {category}")
#     print("========================================")

#     TRAIN_DIR = PROCESSED_DIR / category / "train"
#     VAL_DIR = PROCESSED_DIR / category / "val"

#     train_dataset = SegmentationDataset(
#         TRAIN_DIR,
#         image_transform,
#         mask_transform
#     )

#     val_dataset = SegmentationDataset(
#         VAL_DIR,
#         image_transform,
#         mask_transform
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=0
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=0
#     )

#     model = UNet().to(DEVICE)

#     criterion = nn.BCEWithLogitsLoss()

#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=LEARNING_RATE
#     )

#     for epoch in range(EPOCHS):

#         model.train()

#         train_loss = 0

#         for images, masks in train_loader:

#             images = images.to(DEVICE)
#             masks = masks.to(DEVICE)

#             optimizer.zero_grad()

#             outputs = model(images)

#             loss = criterion(
#                 outputs,
#                 masks
#             )

#             loss.backward()

#             optimizer.step()

#             train_loss += loss.item()

#         model.eval()

#         val_loss = 0

#         with torch.no_grad():

#             for images, masks in val_loader:

#                 images = images.to(DEVICE)
#                 masks = masks.to(DEVICE)

#                 outputs = model(images)

#                 loss = criterion(
#                     outputs,
#                     masks
#                 )

#                 val_loss += loss.item()

#         train_loss /= len(train_loader)
#         val_loss /= len(val_loader)

#         print(
#             f"Epoch [{epoch + 1}/{EPOCHS}] "
#             f"Train Loss: {train_loss:.4f} "
#             f"Val Loss: {val_loss:.4f}"
#         )

#     category_model_dir = (
#         MODEL_DIR /
#         category /
#         "segmentation"
#     )

#     category_model_dir.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     model_path = (
#         category_model_dir /
#         "unet.pth"
#     )

#     torch.save(
#         model.state_dict(),
#         model_path
#     )

#     print(
#         f"\n{category} U-Net saved at:"
#     )

#     print(model_path)

# print("\n========================================")
# print("ALL CATEGORY SEGMENTATION MODELS TRAINED")
# print("========================================")



import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from pathlib import Path
from dataset import SegmentationDataset, image_transform, mask_transform
from unet import UNet

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001

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
    print(f"Training segmentation model: {category}")
    print("========================================")

    TRAIN_DIR = PROCESSED_DIR / category / "train"
    VAL_DIR = PROCESSED_DIR / category / "val"

    train_dataset = SegmentationDataset(
        TRAIN_DIR,
        image_transform,
        mask_transform,
        is_train=True
    )

    val_dataset = SegmentationDataset(
        VAL_DIR,
        image_transform,
        mask_transform
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

    model = UNet().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    masks
                )

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

    category_model_dir = (
        MODEL_DIR /
        category /
        "segmentation"
    )

    category_model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        category_model_dir /
        "unet.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        f"\n{category} U-Net saved at:"
    )

    print(model_path)

print("\n========================================")
print("ALL CATEGORY SEGMENTATION MODELS TRAINED")
print("========================================")