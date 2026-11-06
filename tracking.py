
import cv2
import torch
import time
from ultralytics import YOLO

# 1. Chon thiet bi tinh toan
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Device:", device)

# 2. Tai model YOLO11n
model = YOLO("yolo11n.pt")

# 3. Mo camera MacBook
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Khong mo duoc camera")

# 4. Vong lap realtime
while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc frame")
        break

    start_time = time.perf_counter()

    # 5. YOLO + ByteTrack
    results = model.track(
        frame,
        device=device,
        classes=[0],
        conf=0.25,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    # 6. Ve bounding box va ID
    annotated_frame = result.plot()

    # 7. Lay danh sach track ID
    boxes = result.boxes

    if boxes is not None and boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()

        print("Track IDs:", track_ids)

        current_count = len(track_ids)
    else:
        current_count = 0

    # 8. Tinh FPS cua vong lap
    elapsed = time.perf_counter() - start_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0

    # 9. Hien thi so nguoi va FPS
    cv2.putText(
        annotated_frame,
        f"People: {current_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Processing FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO Tracking", annotated_frame)

    # Nhan Q de thoat
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Tracking stopped")
