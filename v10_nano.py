import cv2
from ultralytics import YOLO

# Load the YOLOv10 model (nano variant here, you can use 'yolov10n.pt', 'yolov10s.pt', etc.)
model = YOLO('yolov10n.pt')

# List of input video file names
video_files = ['1.mp4', '2.mp4', '3.mp4', '4.mp4']
output_video_files = ['output_1.mp4', 'output_2.mp4', 'output_3.mp4', 'output_4.mp4']

# List of video capture and writer objects
caps = []
outs = []

# Constants for distance estimation
FOCAL_LENGTH = 800  # Focal length in pixels (adjust based on your camera's calibration)
KNOWN_HEIGHT = 1.7  # Known height of the object in meters (adjust based on actual object size)

# Function to estimate distance based on bounding box height
def estimate_distance(bbox_height):
    if bbox_height == 0:  # Avoid division by zero
        return float('inf')
    return (KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height

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

    # Apply YOLOv10 model to each frame
    for i, frame in enumerate(frames):
        # YOLOv10 inference
        results = model(frame)

        result_frame_writable = frame.copy()

        # Iterate over the detections for this frame
        for result in results:
            boxes = result.boxes  # Get the bounding boxes
            for box in boxes:
                # Extract box coordinates and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID
                label = f"{model.names[cls]}: {conf:.2f}"

                # Calculate the height of the bounding box in pixels
                bbox_height = y2 - y1

                # Estimate distance to the object
                distance = estimate_distance(bbox_height)
                distance_label = f"Distance: {distance:.2f} meters"

                # Draw bounding boxes and labels
                cv2.rectangle(result_frame_writable, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_frame_writable, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(result_frame_writable, distance_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write the frame with detections and distance estimations to the output video
        outs[i].write(result_frame_writable)

        # Optionally, display the frame with detections (can comment out for non-interactive use)
        cv2.imshow(f'YOLOv10 Video {i+1}', result_frame_writable)

    # Press 'q' to exit the video processing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
for cap, out in zip(caps, outs):
    cap.release()
    out.release()

cv2.destroyAllWindows()
