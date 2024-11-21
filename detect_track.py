from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort
import numpy as np
from collections import deque

def detect_track_predict():
    cap = cv2.VideoCapture(r'blr.mp4')  # for video
    model = YOLO("yolov8n.pt")
    
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", 
                  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
                  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
                  "skis", "snowboard", "sports ball", "kite", "baseball", "bat", "baseball glove", "skateboard", "surfboard", 
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
                  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
                  "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", 
                  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
                  "teddy bear", "hair drier", "toothbrush"]

    # Trackers and prediction deque
    tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
    track_history = deque(maxlen=90)  # to store frame counts for prediction
    
    # Define lanes and counters
    limits = [
        {"name": "A_D", "coords": [(480, 70, 575, 480), (365, 30, 455, 30)], "counter": 0, "ids": set()},
        {"name": "A_F", "coords": [(480, 70, 575, 480), (50, 180, 50, 480)], "counter": 0, "ids": set()},
        {"name": "E_D", "coords": [(190, 55, 70, 200), (365, 30, 455, 30)], "counter": 0, "ids": set()},
        {"name": "E_B", "coords": [(190, 55, 70, 200), (670, 85, 670, 230)], "counter": 0, "ids": set()},
        {"name": "C_B", "coords": [(450, 50, 640, 50), (700, 65, 700, 230)], "counter": 0, "ids": set()}
    ]
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        detections = np.empty((0, 5))

        # Process model results
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                currentClass = classNames[cls]
                
                if currentClass in {"car", "bus", "motorbike", "truck"} and conf > 0.3:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        resultsTracker = tracker.update(detections)

        # Draw and count detections
        for lane in limits:
            for (x1, y1, x2, y2) in lane["coords"]:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

        for result in resultsTracker:
            x1, y1, x2, y2, ID = map(int, result)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f'ID {int(ID)}', (max(0, x1), max(35, y1)), scale=0.8, thickness=2, offset=8)
            
            # Check each lane and update counts
            for lane in limits:
                line_a, line_b = lane["coords"]
                if (
                    (line_a[0] - 10 <= cx <= line_a[2] + 10 or line_b[0] - 10 <= cx <= line_b[2] + 10)
                    and (line_a[1] <= cy <= line_a[3] or line_b[1] <= cy <= line_b[3])
                ):
                    if ID not in lane["ids"]:
                        lane["ids"].add(ID)
                        lane["counter"] += 1
                        cv2.line(img, (line_a[0], line_a[1]), (line_a[2], line_a[3]), (0, 255, 0), 5)
        
        # Display counts for each lane
        counts = [f"{lane['name']}: {lane['counter']}" for lane in limits]
        cvzone.putTextRect(img, " | ".join(counts), (30, 450), scale=1, thickness=2)

        # Display image
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()