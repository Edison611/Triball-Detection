from ultralytics import YOLO

model = YOLO("yolov8m")

model.train(data = "data_cutom.yaml", epochs = 100, batch = 8, img_size = 640, workers=1)