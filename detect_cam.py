import cv2

for i in range(5):  # coba sampai 5 device
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Kamera ditemukan di index {i}")
            cv2.imshow(f"Kamera {i}", frame)
            cv2.waitKey(1000)  # tampilkan 1 detik
        cap.release()

cv2.destroyAllWindows()
