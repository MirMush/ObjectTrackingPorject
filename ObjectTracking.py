from ultralytics import YOLO
import cv2
import numpy as np
import sys

# Load YOLO model for card detection
model = YOLO(r"C:\Users\mirmu\OneDrive\Desktop\AI Assessment2\UnoCardDetectionModel.pt")

TrDict = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
}

# Initialize Tracker
tracker = cv2.TrackerCSRT_create()



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

    # Use YOLO to detect cards in the frame
    results = model(frame)  # Perform inference on the frame
    detections = results[0].boxes  # Extract bounding boxes from results

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

        # Loop over all detected cards in the frame
        for detection in detections:
            # Extract card information (bounding box, label, confidence)
            x1, y1, x2, y2 = detection.xyxy[0].tolist()  # Bounding box coordinates (top-left, bottom-right)
            conf = detection.conf[0]  # Confidence score
            card_label = detection.cls[0]  # Card class label

            if conf > 0.5:  # Only display labels with high confidence
                # Draw bounding box for detected card
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Display card label and confidence
                label_text = f"Card {int(card_label)} ({conf*100:.1f}%)"
                cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(5) & 0xFF

    # Quit when 'q' is pressed
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
