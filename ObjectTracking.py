from ultralytics import YOLO
import cv2
import numpy as np
import sys

# Load YOLO model
model = YOLO("yolov8n.pt")

# OpenCV Tracker Algorithms Dictionary
TrDict = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
}

# Initialize Tracker
tracker = TrDict["csrt"]()

# Load video
video = cv2.VideoCapture(r"C:\Users\mirmu\Downloads\WIN_20241124_18_18_45_Pro.mp4")  # Specify video filepath
ret, frame = video.read()

if not ret:
    print("Error: Unable to load video.")
    sys.exit()

# Select ROI (Region of Interest) to track
cv2.imshow("Frame", frame)
BoundingBox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
tracker.init(frame, BoundingBox)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Update tracker and get bounding box
    success, box = tracker.update(frame)
    if success:
        # Convert bounding box to integers
        x, y, w, h = [int(a) for a in box]

        # Draw the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Calculate center of the bounding box (for x, y coordinates)
        center_x = x + w // 2
        center_y = y + h // 2

        # Display coordinates (x, y)
        text = f"Coordinates: x={center_x}, y={center_y}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(5) & 0xFF

    # Quit when 'q' is pressed
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()