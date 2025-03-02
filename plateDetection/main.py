from ultralytics import YOLO
import cv2

#load models
coco_model=YOLO('yolov8n.pt')
license_plate_detector=YOLO(r"E:\AutoParkAI\plateDetectionTraining\runs\detect\train\weights\last.pt")

#load video
cap = cv2.VideoCapture(r"E:\AutoParkAI\plateDetectionTraining\videoTest.mp4")

#read frames
frame_nmr=-1
ret = True
while ret:
    frame_nmr+=1
    #ret : true if frame is captured
    #frame : numpy array for video frame(image)
    ret,frame=cap.read()
    
    if ret and frame_nmr<10:
        #detect vehicles
        detections=coco_model(frame)[0]
        for detection in detections.boxes.data.tolist():
            print(detection)