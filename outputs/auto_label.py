import os
from ultralytics import YOLO

BASE = "D:/Projects/spyne-ai-clone/datasets/vehicle_dataset"
VEHICLE_CLASSES = [2, 3, 5, 7]

model = YOLO("yolov8n.pt")

for split in ["train", "valid"]:
    img_folder = os.path.join(BASE, split, "exterior")
    label_folder = os.path.join(BASE, split, "labels")
    os.makedirs(label_folder, exist_ok=True)

    if not os.path.exists(img_folder):
        continue

    labeled_count = 0
    no_detection_count = 0

    for fname in os.listdir(img_folder):
        img_path = os.path.join(img_folder, fname)
        results = model(img_path, verbose=False)

        boxes = results[0].boxes
        img_h, img_w = results[0].orig_shape

        label_lines = []
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        label_fname = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(label_folder, label_fname)

        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        if label_lines:
            labeled_count += 1
        else:
            no_detection_count += 1

    print(f"[{split}] Labeled: {labeled_count} | No vehicle detected: {no_detection_count}")

print("\nAuto-labeling done.")