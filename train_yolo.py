from ultralytics import YOLO

# ===== Konfigurasi Dataset dan Model =====
DATASET_PATH = r"D:\Identification-Pen-Yolo-KNN\yolo_dataset\data.yaml"
MODEL = "yolov8n.pt"  # model YOLOv8 Nano (ringan, cepat untuk MVP)
EPOCHS = 50
IMG_SIZE = 640

def train_yolo():
    # Load model YOLOv8
    model = YOLO(MODEL)

    # Train model
    results = model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=16,           # batch size, sesuaikan dengan GPU/CPU
        workers=2,          # threads untuk loading data
        optimizer="Adam",   # bisa SGD atau Adam
        verbose=True
    )

    print("\n✅ Training selesai!")
    print(f"📁 Model terbaik tersimpan di: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_yolo()
