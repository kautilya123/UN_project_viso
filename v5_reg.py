import torch
import cv2
import numpy as np

# Load YOLOv5m model (YOLOv5 'medium' variant)
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Change to 'yolov5m'

# Open the video file or webcam stream
video_path = 'input_video.mp4'  # Replace with your video file or 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file or stream")
    exit()

# Get video properties to create the output video file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to save the output video
output_video_path = 'output_with_detections_yolov5m.mp4'  # Specify output file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output file
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Constants for distance estimation
FOCAL_LENGTH = 800  # Approximate focal length (in pixels, based on your camera calibration)
KNOWN_HEIGHT = 1.7  # Known object height in meters (adjust based on real-world object size)

# Function to estimate distance
def estimate_distance(bbox_height):
    if bbox_height == 0:  # Avoid division by zero
        return float('inf')
    distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height
    return distance

# Loop to process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference on the frame
    results = model(frame)

    # Render results (bounding boxes, labels, confidence scores)
    result_frame = results.render()[0]  # Read-only frame

    # Create a writable copy of the result frame
    result_frame_writable = np.copy(result_frame)

    # Extract bounding box info (x1, y1, x2, y2) and class for each detected object
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection[:6]

        # Calculate the height of the bounding box in pixels
        bbox_height = y2 - y1

        # Estimate distance to the object
        distance = estimate_distance(bbox_height)

        # Prepare the text labels
        object_name = model.names[int(cls)]
        label = f"{object_name}: {conf:.2f}"  # Confidence score
        distance_label = f"Distance: {distance:.2f} meters"  # Distance estimation

        # Draw the confidence score label on the frame
        cv2.putText(result_frame_writable, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw the distance label below the confidence score
        cv2.putText(result_frame_writable, distance_label, (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with detections and distance estimations to the output video
    out.write(result_frame_writable)

    # Display the frame with detections and distance estimations
    cv2.imshow('YOLOv5 Inference with Distance (YOLOv5m)', result_frame_writable)

    # Press 'q' to exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
