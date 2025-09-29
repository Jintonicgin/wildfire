from ultralytics import YOLO
m = YOLO("wildfire/ML/weights/best0.pt")
print(m.model.names, len(m.model.names))