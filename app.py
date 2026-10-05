import threading
import time
import cv2
import numpy as np
import urllib.request
from flask import Flask, render_template, Response, jsonify
from tracker import FoodTracker
from config import PHONE_IP_URL, USE_PHONE_CAM, CAMERA_INDEX

app = Flask(__name__)
tracker = FoodTracker()
latest_detections = []


class VideoStreamBuffer:
    def __init__(self, src):
        self.src = src
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        # Check if using phone URL or local webcam
        self.is_url = isinstance(src, str) and src.startswith("http")
        # For IP Webcam, /shot.jpg is much more reliable than /video
        self.shot_url = src.replace("/video", "/shot.jpg") if self.is_url else None

        self.cap = None
        if not self.is_url:
            self.cap = cv2.VideoCapture(self.src)

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            try:
                if self.is_url:
                    # Direct snapshot fetch: Never buffers, zero lag, never drops
                    req = urllib.request.urlopen(self.shot_url, timeout=2.0)
                    img_arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.frame = frame
                    time.sleep(0.03)  # ~30 FPS polling
                else:
                    if self.cap is None or not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(self.src)
                        time.sleep(0.5)
                        continue
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        with self.lock:
                            self.frame = frame
                    else:
                        time.sleep(0.01)
            except Exception as e:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()


source = PHONE_IP_URL if USE_PHONE_CAM else CAMERA_INDEX
stream_buffer = VideoStreamBuffer(source)


def generate_frames():
    global latest_detections

    while True:
        success, frame = stream_buffer.read()
        if not success or frame is None:
            time.sleep(0.05)
            continue

        try:
            # Process detections & draw overlay
            annotated_frame, detections = tracker.process_frame(frame)
            latest_detections = detections

            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as err:
            print(f"Frame processing error: {err}")
            time.sleep(0.05)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/data')
def get_data():
    return jsonify(latest_detections)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)