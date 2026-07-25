import os
import shutil
import random
from collections import defaultdict

BASE = "D:/Projects/spyne-ai-clone/datasets/angle_dataset_v2"
OUTPUT = "D:/Projects/spyne-ai-clone/datasets/angle_dataset_split"
TRAIN_SPLIT = 0.8

classes = ["front", "side", "rear"]

for split in ["train", "valid"]:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT, split, cls), exist_ok=True)

for cls in classes:
    src_folder = os.path.join(BASE, cls)
    files = os.listdir(src_folder)

    vin_groups = defaultdict(list)
    for fname in files:
        vin = fname.split("_")[0]
        vin_groups[vin].append(fname)

    vins = list(vin_groups.keys())
    random.seed(42)
    random.shuffle(vins)

    split_idx = int(len(vins) * TRAIN_SPLIT)
    train_vins = vins[:split_idx]
    val_vins = vins[split_idx:]

    train_count, val_count = 0, 0
    for vin in train_vins:
        for fname in vin_groups[vin]:
            shutil.copy(os.path.join(src_folder, fname), os.path.join(OUTPUT, "train", cls, fname))
            train_count += 1
    for vin in val_vins:
        for fname in vin_groups[vin]:
            shutil.copy(os.path.join(src_folder, fname), os.path.join(OUTPUT, "valid", cls, fname))
            val_count += 1

    print(f"[{cls}] Train: {train_count} | Valid: {val_count}")

print("\nDone. Dataset ready at:", OUTPUT)