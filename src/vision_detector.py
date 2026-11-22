import cv2
import numpy as np
from ultralytics import YOLO

class TrafficVisionDetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        # COCO vehicle class IDs: car(2), motorcycle(3), bus(5), truck(7)
        self.vehicle_classes = {2: "car", 3: "bike", 5: "bus", 7: "truck"}
        # Weighted congestion score per vehicle type (PCU - Passenger Car Units)
        self.pcu_weights = {"car": 1.0, "bike": 0.5, "bus": 2.5, "truck": 2.5, "ambulance": 10.0}

        # Define polygon coordinates for 4 lanes (Adjust based on your video resolution)
        self.lane_polygons = {
            "North": np.array([[200, 100], [400, 100], [450, 350], [150, 350]], np.int32),
            "South": np.array([[500, 400], [750, 400], [800, 700], [450, 700]], np.int32),
            "East":  np.array([[750, 200], [1050, 200], [1050, 380], [750, 380]], np.int32),
            "West":  np.array([[50, 380], [350, 380], [350, 550], [50, 550]], np.int32),
        }

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)[0]
        lane_data = {
            lane: {"count": 0, "pcu": 0.0, "emergency": False}
            for lane in self.lane_polygons
        }

        # Draw lane ROIs
        for lane_name, poly in self.lane_polygons.items():
            cv2.polylines(frame, [poly], isClosed=True, color=(255, 255, 0), thickness=2)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id not in self.vehicle_classes or conf < 0.4:
                continue

            v_type = self.vehicle_classes[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Custom heuristic/class check for emergency vehicles (if custom model used, check cls_name == 'ambulance')
            is_emergency = False 

            # Check which lane polygon the vehicle center falls into
            for lane_name, poly in self.lane_polygons.items():
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    lane_data[lane_name]["count"] += 1
                    lane_data[lane_name]["pcu"] += self.pcu_weights[v_type]
                    if is_emergency:
                        lane_data[lane_name]["emergency"] = True

                    # Draw bounding box
                    color = (0, 0, 255) if is_emergency else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{v_type}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    break

        return frame, lane_data
