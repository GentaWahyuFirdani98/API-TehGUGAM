from ultralytics import YOLO
import torch
import sys

def train_model():
    # Cek GPU
    if not torch.cuda.is_available():
        print("❌ GPU tidak terdeteksi. Pastikan CUDA dan driver NVIDIA terinstal.")
        sys.exit(1)  # Langsung hentikan eksekusi

    print("✅ GPU tersedia:", torch.cuda.get_device_name(0))

    # Gunakan model YOLOv8 pretrained untuk OBB
    model = YOLO("yolov8n-obb.pt")  # Bisa diganti dengan yolov8s/m/l-obb.pt

    # Jalankan training
    model.train(
        data="tea diseases.v3i.yolov8-obb/data.yaml",  # Sesuaikan dengan path-mu
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,  # Hanya pakai GPU device 0
        name="train-tea-obb"
    )

if __name__ == "__main__":
    train_model()
