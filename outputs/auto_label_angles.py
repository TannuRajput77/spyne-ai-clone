import os
import shutil
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

label_prompts = [
    "exterior view of a whole car outside",
    "interior view inside a car, dashboard or seats",
    "close-up detail shot of a car part like wheel, headlight, or badge"
]
label_names = ["exterior", "interior", "misc"]

SOURCE = "D:/dataset"   # VIN folders are here
OUTPUT = "D:/Projects/spyne-ai-clone/datasets/sorted_by_clip"

for name in label_names:
    os.makedirs(os.path.join(OUTPUT, name), exist_ok=True)

all_image_paths = []
for vin_folder in os.listdir(SOURCE):
    folder_path = os.path.join(SOURCE, vin_folder)
    if not os.path.isdir(folder_path):
        continue
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            all_image_paths.append((vin_folder, fname, os.path.join(folder_path, fname)))

print(f"Found {len(all_image_paths)} total images across all VIN folders.")

for i, (vin_folder, fname, full_path) in enumerate(all_image_paths):
    try:
        image = Image.open(full_path).convert("RGB")
    except Exception:
        continue

    inputs = processor(text=label_prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best_idx = probs.argmax().item()
    best_label = label_names[best_idx]

    new_name = f"{vin_folder}_{fname}"
    shutil.copy(full_path, os.path.join(OUTPUT, best_label, new_name))

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(all_image_paths)}")

print("\nDone. Results:")
for name in label_names:
    folder = os.path.join(OUTPUT, name)
    print(f"{name}: {len(os.listdir(folder))} images")