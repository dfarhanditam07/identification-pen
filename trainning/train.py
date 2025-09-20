import os
import json
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import joblib

# ===== Config =====
BASE_DIR = r"D:\Identification-Pen-Yolo-KNN"
TRAIN_DIR = os.path.join(BASE_DIR, "trainning", "dataset", "train")
VAL_DIR   = os.path.join(BASE_DIR, "trainning", "dataset", "val")
MODELS_DIR = os.path.join(BASE_DIR, "implementation", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = (64, 64)  # resize gambar agar KNN lebih cepat

# ===== Fungsi load dataset =====
def load_dataset(path):
    features, labels, class_names = [], [], []
    for i, class_name in enumerate(sorted(os.listdir(path))):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                try:
                    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                    img_array = np.array(img) / 255.0
                    features.append(img_array.flatten())
                    labels.append(i)
                except Exception as e:
                    print(f"Skipped {img_path}: {e}")
    return np.array(features), np.array(labels), class_names

# ===== Load train & validation =====
train_features, train_labels, class_names = load_dataset(TRAIN_DIR)
val_features, val_labels, _ = load_dataset(VAL_DIR)

# ===== Simpan mapping label =====
idx_to_class = {i: name for i, name in enumerate(class_names)}
with open(os.path.join(MODELS_DIR, "labels_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, indent=2, ensure_ascii=False)

# ===== Normalisasi & scale fitur =====
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled   = scaler.transform(val_features)

# ===== Cari K terbaik dengan cross-validation =====
best_k = None
best_score = 0
print("🔍 Mencari K terbaik (1–10)...")
for k in range(1, 11):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_temp, train_features_scaled, train_labels, cv=5)
    mean_score = scores.mean()
    print(f"   k={k}, mean CV accuracy={mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\n✅ K terbaik = {best_k} dengan mean CV accuracy = {best_score:.4f}")

# ===== Train KNN dengan K terbaik =====
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(train_features_scaled, train_labels)

# ===== Evaluate =====
val_preds = knn.predict(val_features_scaled)
accuracy = np.mean(val_preds == val_labels)
print(f"\n🎯 Validation Accuracy: {accuracy:.4f}")

# Hitung Precision, Recall, F1
precision = precision_score(val_labels, val_preds, average="macro")
recall    = recall_score(val_labels, val_preds, average="macro")
f1        = f1_score(val_labels, val_preds, average="macro")

print(f"📊 Precision: {precision:.4f}")
print(f"📊 Recall:    {recall:.4f}")
print(f"📊 F1-score:  {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(val_labels, val_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Validation Set")
plt.show()

# Classification Report
print("\nClassification Report:\n", classification_report(val_labels, val_preds, target_names=class_names))

# ===== Save model & scaler =====
joblib.dump(knn, os.path.join(MODELS_DIR, "knn_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "knn_scaler.pkl"))

print("✅ Training selesai. Model KNN + Scaler disimpan di:", MODELS_DIR)
