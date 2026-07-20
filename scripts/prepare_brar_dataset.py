import os
import shutil
import random
from pathlib import Path


def prepare_brar_dataset(source_dir, dest_dir, split_ratio=0.8, seed=42):
    random.seed(seed)

    # Map BRAR levels to C-classes
    level_map = {"level_1": "C1", "level_2": "C2", "level_3": "C3"}

    # Create train and test dirs
    train_dir = os.path.join(dest_dir, "train")
    test_dir = os.path.join(dest_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    total_copied = 0

    for level, c_class in level_map.items():
        level_dir = os.path.join(source_dir, level)
        if not os.path.exists(level_dir):
            print(f"Warning: {level_dir} not found.")
            continue

        images = [
            f for f in os.listdir(level_dir) if f.lower().endswith((".jpg", ".png"))
        ]
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # Copy train
        for img in train_images:
            src_path = os.path.join(level_dir, img)
            dst_name = f"{c_class}_{img}"
            dst_path = os.path.join(train_dir, dst_name)
            shutil.copy2(src_path, dst_path)

        # Copy test
        for img in test_images:
            src_path = os.path.join(level_dir, img)
            dst_name = f"{c_class}_{img}"
            dst_path = os.path.join(test_dir, dst_name)
            shutil.copy2(src_path, dst_path)

        print(
            f"{level} -> {c_class}: {len(train_images)} train, {len(test_images)} test"
        )
        total_copied += len(images)

    print(f"Dataset preparation complete. Total images: {total_copied}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare BRAR dataset")
    parser.add_argument(
        "--source",
        type=str,
        default=r"C:\Users\chema\Github\Dental_014\data\BRAR_dataset",
        help="Path to raw BRAR dataset",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=r"C:\Users\chema\Github\Dental_014\data\clean_cropped_ds",
        help="Destination path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    prepare_brar_dataset(args.source, args.dest, seed=args.seed)
