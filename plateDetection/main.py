from ultralytics import YOLO
import cv2
from sort.sort import*
from util import get_car

#load tracker
mot_tracker=Sort()

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
        
        #track vehicles
        track_ids=mot_tracker.update(np.asarray(detections_))

        #detect license plates
        license_plates=license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id=license_plate

            #assign license plate to car 
            xcar1,ycar1,xcar2,ycar2,car_id=get_car(license_plate,track_ids)

            #crop license plate
            license_plate_crop=frame[int(y1):int(y2),int(x1):int(x2),:]

            #process license plate
            license_plate_crop_gray=cv2.cvtColor(license_plate_crop,cv2.COLOR_BGR2GRAY)
            _,license_plate_crop_thresh=cv2.threshold(license_plate_crop_gray,64,255,cv2.THRESH_BINARY_INV)

            cv2.imshow('original_crop',license_plate_crop)
            cv2.imshow('threshold',license_plate_crop_thresh)

            cv2.waitKey(0)
            
