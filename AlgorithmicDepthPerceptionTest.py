import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLO model for object detection
yolo_model = YOLO(r"C:\Users\mirmu\OneDrive\Desktop\AI Assessment2\UnoCardDetectionModel.pt")

# Camera calibration results
camera_matrix = np.array([[640.52, 0, 311.79],
                          [0, 641.20, 247.18],
                          [0, 0, 1]])
focal_length_x = camera_matrix[0, 0]  # f_x from calibration
focal_length_y = camera_matrix[1, 1]  # f_y from calibration
principal_point_x = camera_matrix[0, 2]  # c_x from calibration
principal_point_y = camera_matrix[1, 2]  # c_y from calibration

# Real-world dimensions of a playing card (in cm)
real_card_width = 6.35  # cm
real_card_height = 8.89  # cm

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLO
    results = yolo_model(frame)
    detections = results[0].boxes  # Extract bounding boxes from results

    for detection in detections:
        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # Center of bounding box
        u = x1 + bbox_width // 2  # Pixel x-coordinate
        v = y1 + bbox_height // 2  # Pixel y-coordinate

        # Class name
        class_id = int(detection.cls[0])
        class_name = results[0].names[class_id]

        # Calculate depth using width
        if bbox_width > 0:
            depth_cm_width = (real_card_width * focal_length_x) / bbox_width
        else:
            depth_cm_width = 0

        # Calculate depth using height
        if bbox_height > 0:
            depth_cm_height = (real_card_height * focal_length_y) / bbox_height
        else:
            depth_cm_height = 0

        # Use the average of both depth calculations
        Z = (depth_cm_width + depth_cm_height) / 2

        # Calculate 3D translation (X, Y, Z)
        X = ((u - principal_point_x) * Z) / focal_length_x
        Y = ((v - principal_point_y) * Z) / focal_length_y

        # Calculate rotation angles
        theta_x = np.degrees(np.arctan2(X, Z))  # Horizontal angle
        theta_y = np.degrees(np.arctan2(Y, Z))  # Vertical angle
        theta_rotation = np.degrees(np.arctan2(bbox_height, bbox_width))  # Rotation angle based on bbox aspect ratio

        # Display translation and rotation information
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Class: {class_name}",
            (x1, y1 - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Translation: X={X:.2f}cm, Y={Y:.2f}cm, Z={Z:.2f}cm",
            (x1, y1 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Rotation: theta_x={theta_x:.2f}°, theta_y={theta_y:.2f}°, rotation={theta_rotation:.2f}°",
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )

    # Display frames
    cv2.imshow("Frame", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
