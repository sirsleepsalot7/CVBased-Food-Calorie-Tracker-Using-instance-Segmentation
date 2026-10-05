import cv2
from ultralytics import YOLO
from calibration import ScaleCalibrator
from config import CAMERA_INDEX

# 1. Load weights
model_path = r"runs\segment\food_training_runs\fruits_seg-6\weights\best.pt"
model = YOLO(model_path)
calibrator = ScaleCalibrator()

cap = cv2.VideoCapture(CAMERA_INDEX)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read frame from camera.")
    exit()

print("--- 1. Testing Calibration (Coin Detection) ---")
ppm, pts = calibrator.calculate_ppm(frame)
if ppm:
    print(f"[SUCCESS] Coin found! PPM: {ppm:.2f} px/cm")
else:
    print("[FAILED] Coin not detected. (Check contrast or lighting)")

print("\n--- 2. Testing Model Predictions (Low Conf 0.10) ---")
results = model.predict(frame, conf=0.10, imgsz=640, verbose=False)

if results and len(results) > 0 and len(results[0].boxes) > 0:
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        name = model.names.get(cls_id, "unknown")
        print(f"[DETECTED] Class: '{name}' | Confidence: {conf:.2f}")
else:
    print("[FAILED] Model detected 0 objects even at conf=0.10.")
    print("Falling back to pre-trained base model test (yolov8n-seg.pt)...")
    base_model = YOLO("yolov8n-seg.pt")
    base_res = base_model.predict(frame, conf=0.25, verbose=False)
    for box in base_res[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        print(f"Base YOLOv8 detected: {base_model.names[cls_id]} ({conf:.2f})")