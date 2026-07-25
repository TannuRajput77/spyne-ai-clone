import cv2
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn as nn
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

image_path = "datasets/vehicle_dataset/valid/images/1FTEX1EP0KKD08810_1_3_Exterior_bc2980f590dd40ee80e7fc4ee9339c33.jpg"

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

from PIL import Image
pil_crop = Image.fromarray(cropped)
input_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = classifier(input_tensor)
    probs = torch.softmax(output, dim=1)
    pred_idx = probs.argmax(dim=1).item()
    pred_conf = probs[0][pred_idx].item()

predicted_angle = class_names[pred_idx]
print(f"Predicted angle: {predicted_angle} ({pred_conf:.2%} confidence)")

predictor.set_image(image_rgb)
masks, scores, _ = predictor.predict(box=box, multimask_output=False)
mask = masks[0]
print(f"Segmentation confidence: {scores[0]:.3f}")

overlay = image.copy()
overlay[mask] = [0, 255, 0]
blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.putText(blended, f"{predicted_angle} ({pred_conf:.0%})", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

cv2.imwrite("outputs/pipeline_result.jpg", blended)

mask_uint8 = (mask * 255).astype(np.uint8)
b, g, r = cv2.split(image)
rgba = cv2.merge([b, g, r, mask_uint8])
cv2.imwrite("outputs/pipeline_cutout.png", rgba)

print("Saved: outputs/pipeline_result.jpg")
print("Saved: outputs/pipeline_cutout.png")