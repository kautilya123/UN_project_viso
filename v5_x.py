import torch
import cv2
import numpy as np

# Load YOLOv5l model (YOLOv5 'large' variant)
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

# List of input video file names
video_files = ['input_video.mp4', 'input_video_chair.mp4', 'input_video_chair2.mp4', 'input_video_chair3.mp4']
output_video_files = ['output_1.mp4', 'output_2.mp4', 'output_3.mp4', 'output_4.mp4']

# List of video capture and writer objects
caps = []
outs = []

# Constants for distance estimation
FOCAL_LENGTH = 800  # Approximate focal length (in pixels, based on your camera calibration)
KNOWN_HEIGHT = 1.7  # Known object height in meters (adjust based on real-world object size)

# Function to estimate distance
def estimate_distance(bbox_height):
    if bbox_height == 0:  # Avoid division by zero
        return float('inf')
    distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height
    return distance

# Initialize VideoCapture and VideoWriter for each video
for i in range(len(video_files)):
    cap = cv2.VideoCapture(video_files[i])
    if not cap.isOpened():
        print(f"Error opening video file {video_files[i]}")
        exit()

    # Get video properties to create the output video file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output file
    out = cv2.VideoWriter(output_video_files[i], fourcc, fps, (frame_width, frame_height))

    caps.append(cap)
    outs.append(out)

# Process each video simultaneously
while True:
    frames = []
    rets = []
    
    # Read frames from each video
    for cap in caps:
        ret, frame = cap.read()
        rets.append(ret)
        frames.append(frame)

    # Break the loop if any of the videos end
    if not all(rets):
        break

    # Apply YOLOv5 model to each frame
    for i, frame in enumerate(frames):
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
        outs[i].write(result_frame_writable)

        # Display the frame with detections and distance estimations
        cv2.imshow(f'YOLOv5 Inference Video {i+1}', result_frame_writable)

    # Press 'q' to exit the video processing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
for cap, out in zip(caps, outs):
    cap.release()
    out.release()

cv2.destroyAllWindows()
