import os
import shutil

SOURCE = "D:/dataset/WAUFJBFW7N7006646"
OUTPUT = "D:/Projects/spyne-ai-clone/datasets/sfm_images_v2"

os.makedirs(OUTPUT, exist_ok=True)

count = 0
for fname in os.listdir(SOURCE):
    if "_Exterior_" in fname and fname.lower().endswith((".jpg", ".jpeg", ".png")):
        shutil.copy(os.path.join(SOURCE, fname), os.path.join(OUTPUT, fname))
        count += 1

print(f"Copied {count} exterior images to {OUTPUT}")