from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
from sort import Sort

def detect_and_track(video_path):
    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n.pt")
    tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
    
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]
    traffic_counts = []

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        detections = np.empty((0, 5))
        count = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                currentClass = classNames[cls]

                # Filter for specific classes with confidence threshold
                if currentClass in {"car", "bus", "motorbike", "truck"} and conf > 0.3:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
                    count += 1

        tracker.update(detections)
        traffic_counts.append(count)

    cap.release()
    return traffic_counts