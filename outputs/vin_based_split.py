import os
import shutil
import random
from collections import defaultdict

BASE = "D:/Projects/spyne-ai-clone/datasets/vehicle_dataset"
TRAIN_SPLIT = 0.8

all_exterior_dir = os.path.join(BASE, "all_exterior_temp")
os.makedirs(all_exterior_dir, exist_ok=True)

for split in ["train", "valid"]:
    src = os.path.join(BASE, split, "exterior")
    if os.path.exists(src):
        for fname in os.listdir(src):
            shutil.move(os.path.join(src, fname), os.path.join(all_exterior_dir, fname))

vin_groups = defaultdict(list)
for fname in os.listdir(all_exterior_dir):
    vin = fname.split("_")[0]
    vin_groups[vin].append(fname)

print(f"Total unique VINs: {len(vin_groups)}")

vins = list(vin_groups.keys())
random.seed(42)
random.shuffle(vins)

split_idx = int(len(vins) * TRAIN_SPLIT)
train_vins = vins[:split_idx]
val_vins = vins[split_idx:]

new_train_dir = os.path.join(BASE, "train", "exterior")
new_val_dir = os.path.join(BASE, "valid", "exterior")
os.makedirs(new_train_dir, exist_ok=True)
os.makedirs(new_val_dir, exist_ok=True)

train_count, val_count = 0, 0

for vin in train_vins:
    for fname in vin_groups[vin]:
        shutil.move(os.path.join(all_exterior_dir, fname), os.path.join(new_train_dir, fname))
        train_count += 1

for vin in val_vins:
    for fname in vin_groups[vin]:
        shutil.move(os.path.join(all_exterior_dir, fname), os.path.join(new_val_dir, fname))
        val_count += 1

os.rmdir(all_exterior_dir)

print(f"Train VINs: {len(train_vins)} -> {train_count} images")
print(f"Valid VINs: {len(val_vins)} -> {val_count} images")
print("VIN-based split done. No VIN overlaps between train and valid now.")