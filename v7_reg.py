import cv2
import torch
from utils.general import non_max_suppression
from utils.datasets import letterbox
from utils.plots import plot_one_box
import numpy as np

# Load the YOLOv7 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights = 'yolov7.pt'  # Path to the YOLOv7 model weights
model = torch.load(weights, map_location=device)['model'].float().eval()  # Load the model

# List of input video file names
video_files = ['1.mp4', '2.mp4', '3.mp4', '4.mp4']
output_video_files = ['output_1.mp4', 'output_2.mp4', 'output_3.mp4', 'output_4.mp4']

# Video properties and capture objects
caps = []
outs = []

# Constants for distance estimation
FOCAL_LENGTH = 800  # Approximate focal length (in pixels)
KNOWN_HEIGHT = 1.7  # Known object height in meters (adjust based on object size)

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

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output file
    out = cv2.VideoWriter(output_video_files[i], fourcc, fps, (frame_width, frame_height))

    caps.append(cap)
    outs.append(out)

# Function to process YOLOv7 model inference
def run_yolo_inference(frame, model):
    img = letterbox(frame, new_shape=640)[0]  # Resize the image to 640x640
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)
    
    # Convert to tensor
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # Normalize to 0-1
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # YOLOv7 inference
    with torch.no_grad():
        pred = model(img)[0]
    
    # Apply NMS (non-max suppression)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    return pred

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

    # Apply YOLOv7 model to each frame
    for i, frame in enumerate(frames):
        pred = run_yolo_inference(frame, model)

        result_frame_writable = frame.copy()

        # Process detections
        for det in pred:
            if len(det):
                # Rescale boxes from 640 to original image size
                det[:, :4] = det[:, :4].round()

                for *xyxy, conf, cls in det:
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, result_frame_writable, label=label, color=(255, 0, 0), line_thickness=2)

                    # Estimate distance
                    x1, y1, x2, y2 = map(int, xyxy)
                    bbox_height = y2 - y1
                    distance = estimate_distance(bbox_height)
                    distance_label = f'Distance: {distance:.2f} meters'
                    
                    # Add distance label to the image
                    cv2.putText(result_frame_writable, distance_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame with detections and distance estimations to the output video
        outs[i].write(result_frame_writable)

        # Display the frame with detections and distance estimations
        cv2.imshow(f'YOLOv7 Inference Video {i+1}', result_frame_writable)

    # Press 'q' to exit the video processing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
for cap, out in zip(caps, outs):
    cap.release()
    out.release()

cv2.destroyAllWindows()
