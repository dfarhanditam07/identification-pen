import cv2
import numpy as np
import joblib
import sqlite3
import json   # ✅ Tambahkan ini
from datetime import datetime
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler

# ====== CONFIG ======
YOLO_MODEL_PATH = r"D:\Identification-Pen-Yolo-KNN\runs\detect\train\weights\best.pt"
KNN_MODEL_PATH = r"D:\Identification-Pen-Yolo-KNN\implementation\models\knn_model.pkl"
SCALER_PATH   = r"D:\Identification-Pen-Yolo-KNN\implementation\models\knn_scaler.pkl"
LABELS_PATH   = r"D:\Identification-Pen-Yolo-KNN\implementation\models\labels_mapping.json"
DB_PATH       = "results.db"
IMG_SIZE = (64, 64)

# ====== LOAD YOLO MODEL ======
print("🔄 Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

# ====== LOAD KNN MODEL & SCALER ======
print("🔄 Loading KNN model...")
knn = joblib.load(KNN_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ====== LOAD LABELS MAPPING ====== ✅
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_mapping = json.load(f)


# ====== INIT SQLITE DB ======
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    predicted_class TEXT,
    confidence REAL
)
""")
conn.commit()

# ====== FEATURE EXTRACTOR (simple flatten + normalize) ======
def extract_features(image):
    # Resize + normalize (simple feature extractor untuk KNN)
    resized = cv2.resize(image, IMG_SIZE)
    flattened = resized.flatten() / 255.0
    return flattened

# ====== LIVE CAMERA DETECTION ======
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("❌ Tidak bisa membuka kamera")
    exit()

# ✅ Set resolusi agar tidak letterbox
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✅ Kamera aktif, tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Gagal membaca frame dari kamera.")
        break

    results = yolo_model.predict(frame, conf=0.5, verbose=False)
    annotated_frame = frame.copy()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Crop object pulpen
            pulpen_crop = frame[y1:y2, x1:x2]
            if pulpen_crop.size == 0:
                continue

            # Ekstraksi fitur dan prediksi KNN
            feat = extract_features(pulpen_crop)
            feat_scaled = scaler.transform([feat])

            # Prediksi class (angka) dan mapping ke nama label
            y_pred = knn.predict(feat_scaled)[0]
            predicted_class = labels_mapping.get(str(int(y_pred)), str(int(y_pred)))  # konversi angka -> label string

            # Warna bounding box sesuai hasil klasifikasi
            color = (0, 255, 0) if "Good" in predicted_class else (0, 0, 255)

            # Gambar bounding box + label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{predicted_class}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Titik pitch (center box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(annotated_frame, (cx, cy), 5, color, -1)

            # Logging ke database
            cursor.execute("INSERT INTO detections (timestamp, predicted_class, confidence) VALUES (?, ?, ?)",
                           (datetime.now().isoformat(), predicted_class, conf))
            conn.commit()

    cv2.imshow("Live Pulpen Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
print("🛑 Deteksi dihentikan. Database tersimpan di:", DB_PATH)
