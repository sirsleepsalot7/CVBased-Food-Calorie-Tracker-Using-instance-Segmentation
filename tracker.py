import time
import math
import numpy as np
import cv2
from ultralytics import YOLO
from calibration import ScaleCalibrator
from config import NUTRITION_DB


class FoodTracker:
    def __init__(self):
        self.calibrator = ScaleCalibrator()
        self.model = YOLO("yolov8n-seg.pt")
        
        # Calibration smoothing
        self.ppm_history = []
        self.active_ppm = None
        self.last_marker_time = 0.0
        self.PPM_CACHE_TIMEOUT = 5.0
        
        # Object tracking state: {id: {"centroid": (x,y), "mass": float, "last_seen": float}}
        self.tracked_objects = {}
        self.next_obj_id = 0
        self.alpha = 0.15  # Heavy smoothing factor for rock-solid stability

    def get_smoothed_ppm(self, raw_ppm):
        if raw_ppm is not None and raw_ppm > 0:
            self.ppm_history.append(raw_ppm)
            if len(self.ppm_history) > 15:
                self.ppm_history.pop(0)
            self.active_ppm = float(np.median(self.ppm_history))
            self.last_marker_time = time.time()
            return self.active_ppm
        
        if (time.time() - self.last_marker_time) < self.PPM_CACHE_TIMEOUT:
            return self.active_ppm
        return None

    def estimate_volume(self, mask_pts, ppm, cls_name):
        """
        Calculates volume using food-geometry specific modeling:
        - Elongated items (Banana, Cucumber): Curved cylinder approximation
        - Round items (Apple, Orange, Guava, Tomato): Prolate Spheroid
        """
        area_px = cv2.contourArea(mask_pts)
        if area_px <= 0 or ppm <= 0:
            return 0.0

        area_cm2 = area_px / (ppm ** 2)

        # Bananas & long items: Volume ≈ Area * Average Thickness (thickness ≈ width ≈ 3.2cm)
        if "banana" in cls_name or "cucumber" in cls_name:
            # Fit minimum area rectangle to get length and thickness
            rect = cv2.minAreaRect(mask_pts)
            (w_box, h_box) = rect[1]
            length_px = max(w_box, h_box)
            width_px = min(w_box, h_box)
            
            radius_cm = (width_px / ppm) / 2.0
            # Physical constraint: banana cross-section radius is typically 1.3 - 1.8 cm
            radius_cm = max(1.1, min(radius_cm, 1.85))
            
            length_cm = length_px / ppm
            # Standard cylindrical volume formula with curvature reduction
            volume_cm3 = math.pi * (radius_cm ** 2) * (length_cm * 0.85)
            return volume_cm3

        # Round fruits: Prolate Spheroid
        if len(mask_pts) >= 5:
            ellipse = cv2.fitEllipse(mask_pts)
            (_, (d1, d2), _) = ellipse
            a = (max(d1, d2) / 2.0) / ppm
            b = (min(d1, d2) / 2.0) / ppm
            # Clamp height to width to prevent prolate inflation
            b = min(b, a * 0.95)
            return (4.0 / 3.0) * math.pi * a * (b ** 2)

        # Fallback sphere from area
        r_cm = math.sqrt(area_cm2 / math.pi)
        return (4.0 / 3.0) * math.pi * (r_cm ** 3)

    def match_track_id(self, centroid, current_time):
        """Persistent nearest-neighbor object locking to eliminate flickering."""
        best_id = None
        min_dist = 65  # Pixel threshold for same object

        for obj_id, data in list(self.tracked_objects.items()):
            # Expire tracks not seen for 3 seconds
            if current_time - data["last_seen"] > 3.0:
                del self.tracked_objects[obj_id]
                continue

            dist = math.dist(centroid, data["centroid"])
            if dist < min_dist:
                min_dist = dist
                best_id = obj_id

        if best_id is None:
            best_id = self.next_obj_id
            self.next_obj_id += 1
            self.tracked_objects[best_id] = {"centroid": centroid, "mass": 0.0, "last_seen": current_time}
        else:
            self.tracked_objects[best_id]["centroid"] = centroid
            self.tracked_objects[best_id]["last_seen"] = current_time

        return best_id

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]

        current_time = time.time()
        annotated_frame = frame.copy()

        # 1. Scale Calibration (Filtered Coin Detection)
        raw_ppm, coin_pts = self.calibrator.calculate_ppm(frame)
        active_ppm = self.get_smoothed_ppm(raw_ppm)
        
        coin_center = None
        coin_radius = 0
        if raw_ppm and coin_pts is not None and len(coin_pts) >= 3:
            (cx, cy), coin_radius = cv2.minEnclosingCircle(coin_pts)
            coin_center = (int(cx), int(cy))
            coin_radius = int(coin_radius)
            cv2.circle(annotated_frame, coin_center, coin_radius, (0, 255, 255), 2)
            cv2.putText(annotated_frame, "COIN REF", (coin_center[0] - coin_radius, coin_center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

        # 2. YOLO Segmentation
        results = self.model.predict(frame, conf=0.30, imgsz=480, verbose=False)
        detections = []

        if results and len(results) > 0 and results[0].boxes is not None and results[0].masks is not None:
            boxes = results[0].boxes
            masks = results[0].masks

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_w, box_h = x2 - x1, y2 - y1

                # Filter border reflections & edge artifacts
                if (x1 <= 4 or y1 <= 4 or x2 >= (w - 4) or y2 >= (h - 4)) and (box_w > w * 0.4 or box_h > h * 0.4):
                    continue

                # Filter coin overlap
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                if coin_center and math.dist(centroid, coin_center) < (coin_radius * 2.2):
                    continue

                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = self.model.names.get(cls_id, "").lower().strip()

                mask_pts = masks.xy[i].astype(np.int32)
                if len(mask_pts) < 5:
                    continue

                info = NUTRITION_DB.get(cls_name, {
                    "density_g_cm3": 0.94 if "banana" in cls_name else 0.90,
                    "calories_per_100g": 89 if "banana" in cls_name else 52,
                    "protein_per_100g": 1.1,
                    "carbs_per_100g": 22.8,
                    "fat_per_100g": 0.3
                })

                # Visual Mask Overlay
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [mask_pts], (0, 230, 255) if "banana" in cls_name else (0, 210, 80))
                cv2.addWeighted(overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)
                cv2.polylines(annotated_frame, [mask_pts], True, (0, 255, 0), 2)

                # Consistent object tracking ID
                obj_id = self.match_track_id(centroid, current_time)

                mass_g, calories, protein, carbs, fat = 0.0, 0.0, 0.0, 0.0, 0.0
                if active_ppm:
                    vol = self.estimate_volume(mask_pts, active_ppm, cls_name)
                    raw_mass = vol * info["density_g_cm3"]

                    # Smooth out frame-to-frame fluctuations
                    prev_mass = self.tracked_objects[obj_id]["mass"]
                    if prev_mass <= 1.0:
                        mass_g = raw_mass
                    else:
                        mass_g = (self.alpha * raw_mass) + ((1.0 - self.alpha) * prev_mass)

                    self.tracked_objects[obj_id]["mass"] = mass_g

                    calories = (mass_g / 100.0) * info["calories_per_100g"]
                    protein = (mass_g / 100.0) * info["protein_per_100g"]
                    carbs = (mass_g / 100.0) * info["carbs_per_100g"]
                    fat = (mass_g / 100.0) * info["fat_per_100g"]

                # HUD Tagging
                title = f"{cls_name.upper()} #{obj_id} ({conf:.2f})"
                sub1 = f"{mass_g:.1f}g | {calories:.0f} kcal" if active_ppm else "Calibrating scale..."
                sub2 = f"P:{protein:.1f}g C:{carbs:.1f}g F:{fat:.1f}g" if active_ppm else "Place coin in view"

                base_y = max(35, y1 - 10)
                cv2.putText(annotated_frame, title, (x1, base_y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(annotated_frame, sub1, (x1, base_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
                cv2.putText(annotated_frame, sub2, (x1, base_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

                detections.append({
                    "id": obj_id,
                    "name": cls_name,
                    "confidence": round(conf, 2),
                    "mass_g": round(mass_g, 1),
                    "calories": round(calories, 1),
                    "protein": round(protein, 1),
                    "carbs": round(carbs, 1),
                    "fat": round(fat, 1)
                })

        # Global PPM Status
        status_text = f"PPM: {active_ppm:.1f} (Coin OK)" if active_ppm else "PPM: Searching Coin..."
        cv2.putText(annotated_frame, status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0) if active_ppm else (0, 0, 255), 2)

        return annotated_frame, detections