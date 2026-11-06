
import cv2
import torch
from ultralytics import YOLO

# Kiểm tra GPU trên Mac M1
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

# Nạp mô hình YOLO
model = YOLO("yolo11n.pt")

# Mở camera MacBook
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("Khong mo duoc camera!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc camera")
        break

    # YOLO nhận diện người
    results = model.predict(
        frame,
        device=device,
        classes=[0],
        conf=0.4,
        verbose=False
    )

    # Vẽ bounding box
    frame = results[0].plot()

    # Hiển thị camera
    cv2.imshow("YOLO Detection", frame)

    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
