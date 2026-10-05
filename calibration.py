import cv2
import numpy as np
from config import COIN_DIAMETER_CM

class ScaleCalibrator:
    def __init__(self, real_diameter_cm=COIN_DIAMETER_CM):
        self.real_diameter_cm = real_diameter_cm

    def calculate_ppm(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Gaussian blur to suppress fine paper texture
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=60,
            param1=60,
            param2=32,      # Slightly stricter to avoid table edge hallucinations
            minRadius=10,
            maxRadius=50
        )

        best_circle = None
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Pick a circle that is NOT at the extreme borders of the image
            for (cx, cy, r) in circles:
                if 20 < cx < (w - 20) and 20 < cy < (h - 20):
                    best_circle = (cx, cy, r)
                    break

        # Fallback: Contour thresholding on paper
        if best_circle is None:
            # Threshold darker objects on white paper
            _, thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 100 < area < 4000:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity > 0.72:  # High circularity
                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        if 15 < cx < (w - 15) and 15 < cy < (h - 15):
                            candidates.append((cnt, area, int(cx), int(cy), int(radius)))

            if candidates:
                # Pick the smallest circular object (coin, not fruit)
                candidates.sort(key=lambda x: x[1])
                _, _, cx, cy, r = candidates[0]
                best_circle = (cx, cy, r)

        if best_circle is not None:
            cx, cy, r = best_circle
            diameter_px = 2.0 * float(r)
            ppm = diameter_px / self.real_diameter_cm

            angles = np.linspace(0, 2 * np.pi, 24)
            pts = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1).astype(np.int32)
            return ppm, pts

        return None, None