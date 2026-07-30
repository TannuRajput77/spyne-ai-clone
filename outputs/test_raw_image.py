import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIN_FOLDER = "D:/dataset/1GKS1AKC8HR355271"
all_files = [f for f in os.listdir(VIN_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
image_path = os.path.join(VIN_FOLDER, all_files[0])
print(f"Using image: {image_path}")

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

results = yolo_model(image_path)
boxes = results[0].boxes

if len(boxes) == 0:
    print("No vehicle detected.")
    exit()

box = boxes[0].xyxy[0].cpu().numpy()
conf = boxes[0].conf[0].item()
print(f"Detection confidence: {conf:.3f}")

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
print(f"Predicted angle: {predicted_angle} ({pred_conf:.2%} confidence)")

box_center_x = int((x1 + x2) / 2)
box_center_y = int((y1 + y2) / 2)
input_point = np.array([[box_center_x, box_center_y]])
input_label = np.array([1])

predictor.set_image(image_rgb)
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=box,
    multimask_output=True
)

best_idx = np.argmax(scores)
mask = masks[best_idx]
print(f"Segmentation confidence: {scores[best_idx]:.3f}")

mask_uint8_cc = (mask.astype(np.uint8)) * 255
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8_cc, connectivity=8)

if num_labels > 1:
    target_label = labels[box_center_y, box_center_x]
    if target_label == 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        target_label = 1 + np.argmax(areas)
    mask = (labels == target_label)

overlay = image.copy()
overlay[mask] = [0, 255, 0]
blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.putText(blended, f"{predicted_angle} ({pred_conf:.0%})", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

cv2.imwrite("outputs/raw_test_result.jpg", blended)
print("Saved: outputs/raw_test_result.jpg")
print("Mask shape:", mask.shape, "Mask dtype:", mask.dtype, "Mask unique values:", np.unique(mask))

mask_uint8 = (mask * 255).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
mask_uint8 = cv2.GaussianBlur(mask_uint8, (3, 3), 0)
_, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
mask_uint8 = cv2.GaussianBlur(mask_uint8, (3, 3), 0)

b, g, r = cv2.split(image)
rgba = cv2.merge([b, g, r, mask_uint8])
cv2.imwrite("outputs/raw_test_cutout.png", rgba)
print("Saved: outputs/raw_test_cutout.png")

background = np.full((image.shape[0], image.shape[1], 3), 225, dtype=np.uint8)
alpha_float = mask_uint8.astype(float) / 255.0
composite = background.copy()
for c in range(3):
    composite[:, :, c] = (image[:, :, c] * alpha_float + background[:, :, c] * (1 - alpha_float)).astype(np.uint8)
cv2.imwrite("outputs/raw_test_studio.jpg", composite)
print("Saved: outputs/raw_test_studio.jpg")