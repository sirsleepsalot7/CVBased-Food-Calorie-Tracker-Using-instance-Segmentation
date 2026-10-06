import time
import os
import psutil
import numpy as np
import cv2
import urllib.request
from tracker import FoodTracker
from config import PHONE_IP_URL, USE_PHONE_CAM, CAMERA_INDEX

NUM_WARMUP_FRAMES = 10
NUM_TEST_FRAMES = 50

def run_benchmarks():
    print("=" * 65)
    print("     VFCT: COMPUTATIONAL EFFICIENCY & PIPELINE BENCHMARK")
    print("=" * 65)

    # 1. Model Static Properties
    tracker = FoodTracker()
    model = tracker.model
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    # Weight file size
    model_weight_file = "yolov8n-seg.pt"
    weight_size_mb = os.path.getsize(model_weight_file) / (1024 * 1024) if os.path.exists(model_weight_file) else 0.0

    print(f"[*] Model Architecture:      YOLOv8n-seg")
    print(f"[*] Total Parameters:        {total_params:,} ({total_params / 1e6:.2f} Million)")
    print(f"[*] Trainable Parameters:    {trainable_params:,}")
    print(f"[*] Model File Size:         {weight_size_mb:.2f} MB")
    print(f"[*] Standard GFLOPs:         ~12.6 GFLOPs (at 640x640)")
    print("-" * 65)

    # 2. Camera Ingestion Setup
    is_url = USE_PHONE_CAM and isinstance(PHONE_IP_URL, str) and PHONE_IP_URL.startswith("http")
    shot_url = PHONE_IP_URL.replace("/video", "/shot.jpg") if is_url else None
    
    cap = None
    if not is_url:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    def fetch_single_frame():
        if is_url:
            req = urllib.request.urlopen(shot_url, timeout=2.0)
            img_arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        else:
            ret, frame = cap.read()
            return frame if ret else None

    # Test camera connectivity
    print(f"[*] Connecting to source:    {shot_url if is_url else CAMERA_INDEX}")
    initial_frame = fetch_single_frame()
    if initial_frame is None:
        print("[!] Error: Could not connect to camera source for benchmarking.")
        return
    
    h_in, w_in = initial_frame.shape[:2]
    print(f"[*] Input Resolution:        {w_in}x{h_in}")
    print("-" * 65)

    # 3. Warm-up Phase
    print(f"[*] Running {NUM_WARMUP_FRAMES} warm-up frames to populate caches...")
    for _ in range(NUM_WARMUP_FRAMES):
        f = fetch_single_frame()
        if f is not None:
            _ = tracker.process_frame(f)

    # 4. Latency Profiling Arrays
    t_network = []
    t_calibration = []
    t_inference = []
    t_geometry_render = []
    t_encode = []
    t_end_to_end = []

    process = psutil.Process(os.getpid())
    print(f"[*] Profiling {NUM_TEST_FRAMES} frames...")

    for i in range(NUM_TEST_FRAMES):
        t0 = time.perf_counter()

        # Step A: Ingestion / Network Fetch
        t_start_fetch = time.perf_counter()
        raw_frame = fetch_single_frame()
        t_net = (time.perf_counter() - t_start_fetch) * 1000.0
        if raw_frame is None:
            continue

        # Step B: Frame Resize + Coin Calibration
        t_start_calib = time.perf_counter()
        h, w = raw_frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            proc_frame = cv2.resize(raw_frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            proc_frame = raw_frame
        raw_ppm, coin_pts = tracker.calibrator.calculate_ppm(proc_frame)
        _ = tracker.get_smoothed_ppm(raw_ppm)
        t_cal = (time.perf_counter() - t_start_calib) * 1000.0

        # Step C: YOLO Segmentation Inference
        t_start_inf = time.perf_counter()
        results = tracker.model.predict(proc_frame, conf=0.30, imgsz=480, verbose=False)
        t_inf = (time.perf_counter() - t_start_inf) * 1000.0

        # Step D: Food Tracker Geometry & Post-processing
        t_start_geo = time.perf_counter()
        annotated_frame, _ = tracker.process_frame(raw_frame)
        t_geo = (time.perf_counter() - t_start_geo) * 1000.0 - t_cal - t_inf
        t_geo = max(1.0, t_geo)

        # Step E: Web JPEG Encoding
        t_start_enc = time.perf_counter()
        _, _ = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        t_enc = (time.perf_counter() - t_start_enc) * 1000.0

        t_total = (time.perf_counter() - t0) * 1000.0

        # Store runs
        t_network.append(t_net)
        t_calibration.append(t_cal)
        t_inference.append(t_inf)
        t_geometry_render.append(t_geo)
        t_encode.append(t_enc)
        t_end_to_end.append(t_total)

    if cap:
        cap.release()

    # 5. Resource Consumption
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)

    # 6. Aggregated Benchmark Report
    print("\n" + "=" * 65)
    print("                     BENCHMARK RESULTS")
    print("=" * 65)
    print(f"{'Pipeline Stage':<35} | {'Avg Latency (ms)':<15} | {'Std Dev'}")
    print("-" * 65)
    print(f"{'1. Ingestion / Fetch (/shot.jpg)':<35} | {np.mean(t_network):>8.2f} ms     | ±{np.std(t_network):.2f} ms")
    print(f"{'2. Coin Hough Calibration':<35} | {np.mean(t_calibration):>8.2f} ms     | ±{np.std(t_calibration):.2f} ms")
    print(f"{'3. YOLOv8n-seg Inference (CPU)':<35} | {np.mean(t_inference):>8.2f} ms     | ±{np.std(t_inference):.2f} ms")
    print(f"{'4. Geometric Math & Tracking':<35} | {np.mean(t_geometry_render):>8.2f} ms     | ±{np.std(t_geometry_render):.2f} ms")
    print(f"{'5. MJPEG Encoding (Q=70)':<35} | {np.mean(t_encode):>8.2f} ms     | ±{np.std(t_encode):.2f} ms")
    print("-" * 65)
    
    avg_total_lat = np.mean(t_end_to_end)
    fps_throughput = 1000.0 / avg_total_lat

    print(f"{'Total End-to-End Latency':<35} | {avg_total_lat:>8.2f} ms     | ±{np.std(t_end_to_end):.2f} ms")
    print(f"{'Effective Throughput (FPS)':<35} | {fps_throughput:>8.2f} FPS")
    print(f"{'System Memory (RAM) Consumption':<35} | {ram_usage_mb:>8.2f} MB")
    print("=" * 65)

if __name__ == '__main__':
    run_benchmarks()