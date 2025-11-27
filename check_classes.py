from ultralytics import YOLO

model = YOLO("best.pt")
print("Model classes:", model.names)
