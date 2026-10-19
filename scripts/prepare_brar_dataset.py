import os
import re
import shutil
import random
from collections import defaultdict


def extract_patient_group_key(filename: str) -> str:
    """
    Extract a deterministic patient/subject group key from the image filename
    so that multiple radiographs from the same subject never cross train/val splits.
    Examples:
      - 'patient_image_000001_07f48740.jpg' -> 'patient_image_000001'
      - 'subj123_scan2.png' -> 'subj123'
      - '0042.jpg' -> '0042'
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r"^(patient(?:_image)?_\d+)", stem, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    if "_" in stem:
        return stem.rsplit("_", 1)[0].lower()
    return stem.lower()


def prepare_brar_dataset(source_dir, dest_dir, split_ratio=0.8, seed=42):
    rng = random.Random(seed)

    # Map BRAR levels to C-classes
    level_map = {"level_1": "C1", "level_2": "C2", "level_3": "C3"}

    # Create train and test dirs
    train_dir = os.path.join(dest_dir, "train")
    test_dir = os.path.join(dest_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(
        f"[Split Policy] Unit=patient_group_key (80/20 group-aware split, seed={seed})"
    )

    # First pass: collect all patient group keys globally so a patient with images
    # in multiple folders never crosses train/test boundaries across levels either.
    global_groups = defaultdict(list)
    level_entries = defaultdict(list)

    for level, c_class in level_map.items():
        level_dir = os.path.join(source_dir, level)
        if not os.path.exists(level_dir):
            print(f"Warning: {level_dir} not found.")
            continue

        images = sorted(
            [
                f
                for f in os.listdir(level_dir)
                if f.lower().endswith((".jpg", ".png"))
            ]
        )
        for img in images:
            g_key = extract_patient_group_key(img)
            global_groups[g_key].append((level, img))
            level_entries[level].append((g_key, img))

    total_copied = 0
    assigned_split = {}

    for level, c_class in level_map.items():
        entries = level_entries.get(level, [])
        if not entries:
            continue

        level_dir = os.path.join(source_dir, level)
        # Group images by patient key within this level
        grouped = defaultdict(list)
        for g_key, img in entries:
            grouped[g_key].append(img)

        group_keys = sorted(grouped.keys())
        rng.shuffle(group_keys)

        target_train_count = int(len(entries) * split_ratio)
        train_images = []
        test_images = []
        train_groups = 0
        test_groups = 0

        for g_key in group_keys:
            imgs = grouped[g_key]
            if g_key in assigned_split:
                split_choice = assigned_split[g_key]
            elif len(train_images) < target_train_count:
                split_choice = "train"
                assigned_split[g_key] = "train"
            else:
                split_choice = "test"
                assigned_split[g_key] = "test"

            if split_choice == "train":
                train_images.extend(imgs)
                train_groups += 1
            else:
                test_images.extend(imgs)
                test_groups += 1

        # Verify zero patient group overlap between train and test
        train_group_set = {extract_patient_group_key(i) for i in train_images}
        test_group_set = {extract_patient_group_key(i) for i in test_images}
        assert train_group_set.isdisjoint(test_group_set), (
            f"Patient group leakage detected in {level}!"
        )

        for img in train_images:
            src_path = os.path.join(level_dir, img)
            dst_name = f"{c_class}_{img}"
            dst_path = os.path.join(train_dir, dst_name)
            shutil.copy2(src_path, dst_path)

        for img in test_images:
            src_path = os.path.join(level_dir, img)
            dst_name = f"{c_class}_{img}"
            dst_path = os.path.join(test_dir, dst_name)
            shutil.copy2(src_path, dst_path)

        print(
            f"{level} -> {c_class}: {len(train_images)} train ({train_groups} patient groups), "
            f"{len(test_images)} test ({test_groups} patient groups)"
        )
        total_copied += len(entries)

    print(
        f"Dataset preparation complete. Total images: {total_copied}, "
        f"Total unique patient groups: {len(global_groups)}"
    )


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

