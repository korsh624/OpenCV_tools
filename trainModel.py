from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(
    data='ex/dataset/data.yaml',
    epochs=100,
    imgsz=640
)