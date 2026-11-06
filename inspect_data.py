
import cv2
import torch
from ultralytics import YOLO

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO("yolo11n.pt")

# Camera của bạn hoạt động ở index 1
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Khong mo duoc camera")

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Khong doc duoc frame")

# 1. Dữ liệu camera
print("\n=== CAMERA INPUT ===")
print("Type:", type(frame))
print("Shape:", frame.shape)
print("Dtype:", frame.dtype)
print("Pixel [100,200]:", frame[100, 200])

# 2. Dự đoán YOLO
results = model.predict(
    frame,
    imgsz=640,
    device=device,
    conf=0.4,
    classes=[0],
    verbose=False
)

# 3. Tensor sau tiền xử lý
# Xem thông qua predictor của Ultralytics
tensor = model.predictor.preprocess([frame])

print("\n=== MODEL INPUT ===")
print("Type:", type(tensor))
print("Shape:", tuple(tensor.shape))
print("Dtype:", tensor.dtype)
print("Device:", tensor.device)
print("Min:", tensor.min().item())
print("Max:", tensor.max().item())

# 4. Kết quả sau hậu xử lý
result = results[0]

print("\n=== MODEL RESULTS ===")
print("Result type:", type(result))
print("Original shape:", result.orig_shape)

boxes = result.boxes

print("Boxes type:", type(boxes))
print("Boxes shape:", tuple(boxes.data.shape))
print("Boxes data:")
print(boxes.data.cpu().numpy())

print("\nClass names:", result.names)
print("Speed:", result.speed)

# Lưu ảnh để kiểm tra
cv2.imwrite("camera_input.jpg", frame)
result.save(filename="yolo_result.jpg")

print("\nSaved camera_input.jpg and yolo_result.jpg")
