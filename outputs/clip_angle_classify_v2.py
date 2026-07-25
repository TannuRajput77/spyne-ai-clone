import os
import shutil
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Simplified 3 angle categories for better class balance
label_prompts = [
    "front view or front angled view of a car",
    "side profile view of a car",
    "rear view or rear angled view of a car"
]
label_names = ["front", "side", "rear"]

SOURCE = "D:/Projects/spyne-ai-clone/datasets/sorted_by_clip/exterior"
OUTPUT = "D:/Projects/spyne-ai-clone/datasets/angle_dataset_v2"

for name in label_names:
    os.makedirs(os.path.join(OUTPUT, name), exist_ok=True)

files = os.listdir(SOURCE)
print(f"Processing {len(files)} exterior images for angle classification...")

for i, fname in enumerate(files):
    path = os.path.join(SOURCE, fname)
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        continue

    inputs = processor(text=label_prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best_idx = probs.argmax().item()
    best_label = label_names[best_idx]

    shutil.copy(path, os.path.join(OUTPUT, best_label, fname))

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(files)}")

print("\nDone. Angle classification results:")
for name in label_names:
    folder = os.path.join(OUTPUT, name)
    print(f"{name}: {len(os.listdir(folder))} images")