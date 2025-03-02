from ultralytics import YOLO
import cv2

#load models
coco_model=YOLO('yolov8n.pt')
license_plate_detector=YOLO(r"E:\AutoParkAI\plateDetectionTraining\runs\detect\train\weights\last.pt")

#load video
cap = cv2.VideoCapture(r"E:\AutoParkAI\plateDetectionTraining\videoTest.mp4")

#make an array of class ID's (car , motorbike , bus , truck)
vehicles=[2,3,5,7]

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
        detections_=[]
        for detection in detections.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id=detection
            if int(class_id) in vehicles:
                detections_.append([x1,y1,x2,y2,score])
