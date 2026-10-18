# # from pathlib import Path
# # import shutil
# # import random

# # ROOT = Path(__file__).resolve().parents[2]

# # RAW_DIR = ROOT / "data" / "raw" / "mvtec" / "bottle"
# # PROCESSED_DIR = ROOT / "data" / "processed"

# # TRAIN_RATIO = 0.70
# # VAL_RATIO = 0.15
# # SEED = 42

# # random.seed(SEED)


# # def create_folders():
# #     for split in ["train", "val", "test"]:
# #         (PROCESSED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
# #         (PROCESSED_DIR / split / "masks").mkdir(parents=True, exist_ok=True)


# # def collect_data():
# #     data = []

# #     test_dir = RAW_DIR / "test"
# #     ground_truth_dir = RAW_DIR / "ground_truth"

# #     for defect_type in test_dir.iterdir():
# #         if not defect_type.is_dir():
# #             continue

# #         for image_path in defect_type.glob("*"):
# #             if defect_type.name == "good":
# #                 mask_path = None
# #                 label = 0
# #             else:
# #                 mask_path = ground_truth_dir / defect_type.name / f"{image_path.stem}_mask.png"
# #                 label = 1

# #             data.append((image_path, mask_path, label))

# #     return data


# # def split_data(data):
# #     random.shuffle(data)

# #     n = len(data)

# #     train_end = int(n * TRAIN_RATIO)
# #     val_end = train_end + int(n * VAL_RATIO)

# #     train_data = data[:train_end]
# #     val_data = data[train_end:val_end]
# #     test_data = data[val_end:]

# #     return train_data, val_data, test_data


# # def save_data(data, split):
# #     image_dir = PROCESSED_DIR / split / "images"
# #     mask_dir = PROCESSED_DIR / split / "masks"

# #     for i, (image_path, mask_path, label) in enumerate(data):

# #         new_image_name = f"{i:05d}.png"

# #         shutil.copy2(
# #             image_path,
# #             image_dir / new_image_name
# #         )

# #         if mask_path is not None and mask_path.exists():
# #             shutil.copy2(
# #                 mask_path,
# #                 mask_dir / new_image_name
# #             )
# #         else:
# #             from PIL import Image

# #             image = Image.open(image_path)
# #             width, height = image.size

# #             mask = Image.new("L", (width, height), 0)
# #             mask.save(mask_dir / new_image_name)


# # def main():
# #     print("Starting preprocessing...")

# #     create_folders()

# #     data = collect_data()

# #     print("Total images:", len(data))

# #     train_data, val_data, test_data = split_data(data)

# #     print("Train:", len(train_data))
# #     print("Validation:", len(val_data))
# #     print("Test:", len(test_data))

# #     save_data(train_data, "train")
# #     save_data(val_data, "val")
# #     save_data(test_data, "test")

# #     print("\nPreprocessing completed.")


# # if __name__ == "__main__":
# #     main()


# from pathlib import Path
# from PIL import Image
# import shutil
# import random

# ROOT = Path(__file__).resolve().parents[2]

# RAW_DIR = ROOT / "data" / "raw" / "mvtec" / "bottle"
# PROCESSED_DIR = ROOT / "data" / "processed"

# SEED = 42
# random.seed(SEED)


# def create_folders():
#     for split in ["train", "val", "test"]:
#         (PROCESSED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
#         (PROCESSED_DIR / split / "masks").mkdir(parents=True, exist_ok=True)


# def save_image(image_path, mask_path, split, name):
#     image_dir = PROCESSED_DIR / split / "images"
#     mask_dir = PROCESSED_DIR / split / "masks"

#     shutil.copy2(
#         image_path,
#         image_dir / name
#     )

#     if mask_path is not None:
#         shutil.copy2(
#             mask_path,
#             mask_dir / name
#         )
#     else:
#         image = Image.open(image_path)
#         width, height = image.size

#         mask = Image.new(
#             "L",
#             (width, height),
#             0
#         )

#         mask.save(mask_dir / name)


# def split_data(data, train_ratio=0.70, val_ratio=0.15):
#     random.shuffle(data)

#     n = len(data)

#     train_end = int(n * train_ratio)
#     val_end = train_end + int(n * val_ratio)

#     return (
#         data[:train_end],
#         data[train_end:val_end],
#         data[val_end:]
#     )


# def process_good_images():
#     train_good = list(
#         (RAW_DIR / "train" / "good").glob("*.png")
#     )

#     test_good = list(
#         (RAW_DIR / "test" / "good").glob("*.png")
#     )

#     random.shuffle(train_good)

#     val_count = int(len(train_good) * 0.2)

#     val_good = train_good[:val_count]
#     train_good = train_good[val_count:]

#     for i, image_path in enumerate(train_good):
#         save_image(
#             image_path,
#             None,
#             "train",
#             f"good_train_{i:03d}.png"
#         )

#     for i, image_path in enumerate(val_good):
#         save_image(
#             image_path,
#             None,
#             "val",
#             f"good_val_{i:03d}.png"
#         )

#     for i, image_path in enumerate(test_good):
#         save_image(
#             image_path,
#             None,
#             "test",
#             f"good_test_{i:03d}.png"
#         )

#     print("Good images:")
#     print("Train:", len(train_good))
#     print("Val:", len(val_good))
#     print("Test:", len(test_good))


# def process_defect_images():
#     test_dir = RAW_DIR / "test"
#     ground_truth_dir = RAW_DIR / "ground_truth"

#     defect_types = [
#         folder for folder in test_dir.iterdir()
#         if folder.is_dir() and folder.name != "good"
#     ]

#     all_train = []
#     all_val = []
#     all_test = []

#     for defect_type in defect_types:

#         defect_name = defect_type.name

#         images = sorted(
#             defect_type.glob("*.png")
#         )

#         data = []

#         for image_path in images:

#             mask_path = (
#                 ground_truth_dir /
#                 defect_name /
#                 f"{image_path.stem}_mask.png"
#             )

#             if mask_path.exists():
#                 data.append(
#                     (image_path, mask_path, defect_name)
#                 )

#         train_data, val_data, test_data = split_data(data)

#         all_train.extend(train_data)
#         all_val.extend(val_data)
#         all_test.extend(test_data)

#         print(
#             f"{defect_name}: "
#             f"{len(data)} total | "
#             f"{len(train_data)} train | "
#             f"{len(val_data)} val | "
#             f"{len(test_data)} test"
#         )

#     for i, (image_path, mask_path, defect_name) in enumerate(all_train):

#         save_image(
#             image_path,
#             mask_path,
#             "train",
#             f"{defect_name}_{i:03d}.png"
#         )

#     for i, (image_path, mask_path, defect_name) in enumerate(all_val):

#         save_image(
#             image_path,
#             mask_path,
#             "val",
#             f"{defect_name}_{i:03d}.png"
#         )

#     for i, (image_path, mask_path, defect_name) in enumerate(all_test):

#         save_image(
#             image_path,
#             mask_path,
#             "test",
#             f"{defect_name}_{i:03d}.png"
#         )

#     print("\nDefect images:")
#     print("Train:", len(all_train))
#     print("Val:", len(all_val))
#     print("Test:", len(all_test))


# def main():

#     print("Starting MVTec preprocessing...\n")

#     create_folders()

#     process_good_images()

#     print()

#     process_defect_images()

#     print("\nPreprocessing completed successfully.")


# if __name__ == "__main__":
#     main()


from pathlib import Path
from PIL import Image
import shutil
import random

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "mvtec"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
random.seed(SEED)


def create_folders(category):
    for split in ["train", "val", "test"]:
        (PROCESSED_DIR / category / split / "images").mkdir(
            parents=True,
            exist_ok=True
        )

        (PROCESSED_DIR / category / split / "masks").mkdir(
            parents=True,
            exist_ok=True
        )


def save_image(image_path, mask_path, category, split, name):

    image_dir = (
        PROCESSED_DIR /
        category /
        split /
        "images"
    )

    mask_dir = (
        PROCESSED_DIR /
        category /
        split /
        "masks"
    )

    shutil.copy2(
        image_path,
        image_dir / name
    )

    if mask_path is not None:
        shutil.copy2(
            mask_path,
            mask_dir / name
        )

    else:
        image = Image.open(image_path)

        width, height = image.size

        mask = Image.new(
            "L",
            (width, height),
            0
        )

        mask.save(
            mask_dir / name
        )


def split_data(data, train_ratio=0.70, val_ratio=0.15):

    random.shuffle(data)

    n = len(data)

    train_end = int(n * train_ratio)

    val_end = (
        train_end +
        int(n * val_ratio)
    )

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def process_category(category):

    print(f"\n========== {category.upper()} ==========")

    category_dir = RAW_DIR / category

    create_folders(category)

    train_good_dir = (
        category_dir /
        "train" /
        "good"
    )

    test_good_dir = (
        category_dir /
        "test" /
        "good"
    )

    ground_truth_dir = (
        category_dir /
        "ground_truth"
    )

    train_good = list(
        train_good_dir.glob("*.png")
    )

    test_good = list(
        test_good_dir.glob("*.png")
    )

    random.shuffle(train_good)

    val_count = int(
        len(train_good) * 0.20
    )

    val_good = train_good[:val_count]
    train_good = train_good[val_count:]

    for i, image_path in enumerate(train_good):

        save_image(
            image_path,
            None,
            category,
            "train",
            f"good_train_{i:03d}.png"
        )

    for i, image_path in enumerate(val_good):

        save_image(
            image_path,
            None,
            category,
            "val",
            f"good_val_{i:03d}.png"
        )

    for i, image_path in enumerate(test_good):

        save_image(
            image_path,
            None,
            category,
            "test",
            f"good_test_{i:03d}.png"
        )

    test_dir = category_dir / "test"

    defect_types = [
        folder
        for folder in test_dir.iterdir()
        if folder.is_dir()
        and folder.name != "good"
    ]

    total_train = 0
    total_val = 0
    total_test = 0

    for defect_type in defect_types:

        defect_name = defect_type.name

        images = sorted(
            defect_type.glob("*.png")
        )

        data = []

        for image_path in images:

            mask_path = (
                ground_truth_dir /
                defect_name /
                f"{image_path.stem}_mask.png"
            )

            if mask_path.exists():

                data.append(
                    (
                        image_path,
                        mask_path
                    )
                )

        train_data, val_data, test_data = split_data(data)

        for i, (image_path, mask_path) in enumerate(train_data):

            save_image(
                image_path,
                mask_path,
                category,
                "train",
                f"{defect_name}_{i:03d}.png"
            )

        for i, (image_path, mask_path) in enumerate(val_data):

            save_image(
                image_path,
                mask_path,
                category,
                "val",
                f"{defect_name}_{i:03d}.png"
            )

        for i, (image_path, mask_path) in enumerate(test_data):

            save_image(
                image_path,
                mask_path,
                category,
                "test",
                f"{defect_name}_{i:03d}.png"
            )

        total_train += len(train_data)
        total_val += len(val_data)
        total_test += len(test_data)

        print(
            f"{defect_name}: "
            f"{len(data)} total | "
            f"{len(train_data)} train | "
            f"{len(val_data)} val | "
            f"{len(test_data)} test"
        )

    print(
        f"Good: "
        f"{len(train_good)} train | "
        f"{len(val_good)} val | "
        f"{len(test_good)} test"
    )

    print(
        f"Defect: "
        f"{total_train} train | "
        f"{total_val} val | "
        f"{total_test} test"
    )


def main():

    print("Starting MVTec preprocessing...\n")

    categories = [
        folder.name
        for folder in RAW_DIR.iterdir()
        if folder.is_dir()
    ]

    categories.sort()

    print("Categories found:")

    for category in categories:
        print("-", category)

    for category in categories:
        process_category(category)

    print("\n================================")
    print("All categories processed.")
    print("================================")


if __name__ == "__main__":
    main()