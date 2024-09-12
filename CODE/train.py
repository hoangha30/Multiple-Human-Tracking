from ultralytics import YOLO
from ultralytics import settings

model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    settings.reset()
    model = YOLO('yolov8n.pt')
    results = model.train(data="/mnt/HDD/Project/ML2/yolo/dataset.yaml", epochs=100, imgsz=640, batch=16, device=0)
    model.save(save_dir='./weights', model_name='yolov8n_pretrained.pt') 

