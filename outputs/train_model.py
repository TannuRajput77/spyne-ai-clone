from ultralytics import YOLO

# Using YOLO11 instead of YOLOv8 (latest model )
model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=5,
    imgsz=640,
    batch=16,
    name="vehicle_detection_v2"   
)