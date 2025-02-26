import os
import cv2
from ultralytics import YOLO

# Define video paths
video_path = r"E:\AutoParkAI\plateDetectionTraining\videoTest.mp4"  # Ensure the correct file extension
video_path_out = f"{os.path.splitext(video_path)[0]}_out.mp4"  # Output file

# Load YOLO model
model_path = r"E:\AutoParkAI\plateDetectionTraining\runs\detect\train\weights\last.pt"
model = YOLO(model_path)  # Load the trained model

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

# Get video properties
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame. Check your video file.")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

threshold = 0.5  # Confidence threshold for object detection

while ret:
    results = model(frame)[0]  # Perform YOLOv8 inference
    
    if len(results.boxes) == 0:
        print("❌ No objects detected in this frame")

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        print(f"✅ Detected: {results.names[int(class_id)]} | Confidence: {score}")

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)  # Save the processed frame to the output video
    ret, frame = cap.read()  # Read the next frame

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Video saved at {video_path_out}")
