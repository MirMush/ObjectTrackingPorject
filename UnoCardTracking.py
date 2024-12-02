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
camera_index = 0  # Default camera
video = cv2.VideoCapture(camera_index)
if not video.isOpened():
    print("Error: Unable to access the camera.")
    sys.exit()

def get_detections(frame):
    """Use YOLO to detect objects and return bounding boxes."""
    results = model(frame)  # Detect objects
    bboxes = []
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result.tolist()
        bboxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))  # Convert to x, y, w, h
    return bboxes

def draw_bboxes(frame, bboxes):
    """Draw bounding boxes on the frame."""
    for i, (x, y, w, h) in enumerate(bboxes):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Object {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Tracking state
selected_bboxes = []

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Detect objects using YOLO
    detections = get_detections(frame)

    # Draw bounding boxes
    draw_bboxes(frame, detections)

    # Allow selection of objects to track
    if cv2.waitKey(1) & 0xFF == ord("s"):
        print("Select objects to track:")
        for i, bbox in enumerate(detections):
            print(f"{i + 1}: {bbox}")
        selected_indices = input("Enter indices of objects to track (comma-separated): ")
        selected_indices = [int(idx.strip()) - 1 for idx in selected_indices.split(",")]

        # Add selected objects to trackers
        for idx in selected_indices:
            x, y, w, h = detections[idx]
            tracker = TrDict["csrt"]()
            trackers.add(tracker, frame, (x, y, w, h))

        selected_bboxes = [detections[idx] for idx in selected_indices]

    # Update trackers if active
    if selected_bboxes:
        success, boxes = trackers.update(frame)
        for box in boxes:
            x, y, w, h = [int(a) for a in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# Release camera and close windows
video.release()
cv2.destroyAllWindows()
