from ultralytics import YOLO
import torch
  # Harusnya True kalau GPU terdeteksi
print(torch.cuda.is_available())  # Harusnya True
#  # Harusnya "NVIDIA GeForce RTX 3050"


# Cek apakah GPU tersedia
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolov8n.yaml")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        fraction=1.0,
        device=device
    )

if __name__ == "__main__":
    train_model()