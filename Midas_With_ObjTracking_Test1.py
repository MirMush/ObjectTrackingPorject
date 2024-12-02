import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load MiDaS model for depth estimation
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform

# Load YOLO model for object detection
yolo_model = YOLO(r"C:\Users\mirmu\OneDrive\Desktop\AI Assessment2\UnoCardDetectionModel.pt")

# Initialize CSRT Tracker
tracker = cv2.TrackerCSRT_create()

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize variables
tracking = False  # Flag to indicate if tracking is active
bounding_box = None  # Stores the bounding box for tracking
tracked_class_name = None  # Class name of the tracked object

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform depth estimation using MiDaS
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(input_img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)

    # Perform object detection with YOLO
    results = yolo_model(frame)
    detections = results[0].boxes  # Extract bounding boxes from results

    if not tracking:
        # Display detection results
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
            conf = detection.conf[0]  # Confidence score
            if conf > 0.5:  # Display high-confidence detections
                label = int(detection.cls[0])
                class_name = results[0].names[label]  # Get class name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Select the first detected object to track
                if not tracking:
                    bounding_box = (x1, y1, x2 - x1, y2 - y1)
                    tracker.init(frame, bounding_box)
                    tracked_class_name = class_name
                    tracking = True
                    break
    else:
        # Update tracker
        success, bounding_box = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bounding_box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            center_x = x + w // 2
            center_y = y + h // 2

            # Extract depth information for the tracked object
            z_depth = depth_map_normalized[center_y, center_x] * 100  # Convert depth to centimeters

            # Display coordinates and class name in cm
            cv2.putText(
                frame,
                f"Class: {tracked_class_name} | x: {center_x}cm, y: {center_y}cm, z: {z_depth:.2f}cm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    # Display frames
    cv2.imshow("Frame", frame)
    cv2.imshow("Depth Map", (depth_map_normalized * 255).astype(np.uint8))

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
