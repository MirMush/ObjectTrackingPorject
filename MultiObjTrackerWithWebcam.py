from ultralytics import YOLO
import cv2
import numpy as np
import sys

# Load YOLO model
model = YOLO("yolov8n.pt")

# OpenCV Tracker Algorithms Dictionary
TrDict = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
}

# Initialize MultiTracker
trackers = cv2.legacy.MultiTracker_create()

# Open camera
camera_index = 0  # Use 0 for the default camera
video = cv2.VideoCapture(camera_index)
if not video.isOpened():
    print("Error: Unable to access the camera.")
    sys.exit()

# Read the first frame from the camera
ret, frame = video.read()
if not ret:
    print("Error: Unable to capture the camera frame.")
    sys.exit()

# Number of objects to track
obj = 1

# Select ROI for each object and add to MultiTracker
for a in range(obj):
    cv2.imshow("Frame", frame)
    BoundingBox_a = cv2.selectROI("Frame", frame)
    tracker_a = TrDict["csrt"]()  # Initialize tracker
    trackers.add(tracker_a, frame, BoundingBox_a)

cv2.destroyWindow("Frame")  # Close ROI selection window

# Start the tracking loop
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Update tracker and get bounding boxes
    success, boxes = trackers.update(frame)

    # Draw bounding boxes on the frame
    for box in boxes:
        (x, y, w, h) = [int(a) for a in box]  # Convert bounding box to integers
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display tracking text
    text = f"Tracking {len(boxes)} objects"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(5) & 0xFF

    # Quit when 'q' is pressed
    if key == ord("q"):
        break

# Release the camera and close all windows
video.release()
cv2.destroyAllWindows()
