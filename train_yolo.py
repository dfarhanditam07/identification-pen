from ultralytics import YOLO

# ===== Konfigurasi Dataset dan Model =====
DATASET_PATH = r"D:\Identification-Pen-Yolo-KNN\yolo_dataset\data.yaml"
MODEL = "yolov8n.pt"   # model YOLOv8 Nano (ringan, cepat untuk MVP)
EPOCHS = 10
IMG_SIZE = 640

# Folder hasil training
PROJECT_DIR = r"D:\Identification-Pen-Yolo-KNN\implementation\models\yolo"
RUN_NAME = "pen_detector"   # nama folder untuk hasil training

def train_yolo():
    # Load model YOLOv8
    model = YOLO(MODEL)

    # Train model
    results = model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=16,
        workers=2,
        optimizer="Adam",
        project=PROJECT_DIR,   # lokasi folder utama
        name=RUN_NAME,         # nama subfolder run
        verbose=True
    )

    print("\n✅ Training selesai!")
    print(f"📁 Model terbaik tersimpan di: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_yolo()
