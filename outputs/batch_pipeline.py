import os
import random
import cv2
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = YOLO("runs/detect/vehicle_detection_v2/weights/best.pt")

sam_checkpoint = "models/sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

classifier = models.resnet18(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 3)
classifier.load_state_dict(torch.load("models/angle_classifier.pt", map_location=DEVICE))
classifier.to(DEVICE)
classifier.eval()

class_names = ["front", "rear", "side"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

SOURCE_DIR = "datasets/vehicle_dataset/valid/images"
OUTPUT_DIR = "outputs/batch_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_files = os.listdir(SOURCE_DIR)
random.seed(1)
sample_files = random.sample(all_files, min(6, len(all_files)))

for idx, fname in enumerate(sample_files):
    image_path = os.path.join(SOURCE_DIR, fname)

    results = yolo_model(image_path, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print(f"[{idx+1}] {fname} -> No vehicle detected, skipping")
        continue

    box = boxes[0].xyxy[0].cpu().numpy()
    conf = boxes[0].conf[0].item()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x1, y1, x2, y2 = box.astype(int)
    cropped = image_rgb[y1:y2, x1:x2]

    pil_crop = Image.fromarray(cropped)
    input_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = classifier(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_conf = probs[0][pred_idx].item()

    predicted_angle = class_names[pred_idx]

    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    mask = masks[0]

    overlay = image.copy()
    overlay[mask] = [0, 255, 0]
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(blended, f"{predicted_angle} ({pred_conf:.0%})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    out_path = os.path.join(OUTPUT_DIR, f"result_{idx+1}.jpg")
    cv2.imwrite(out_path, blended)

    print(f"[{idx+1}] {fname}")
    print(f"    Detection: {conf:.3f} | Angle: {predicted_angle} ({pred_conf:.2%}) | Segmentation: {scores[0]:.3f}")

print(f"\nDone. Results saved in {OUTPUT_DIR}/")