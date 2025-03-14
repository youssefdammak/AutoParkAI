from ultralytics import YOLO
import cv2
from sort.sort import*
from util import*

results={}

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
    
    if ret:
        results[frame_nmr]={}
        #detect vehicles
        detections=coco_model(frame)[0]
        detections_=[]
        for detection in detections.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id=detection
            if int(class_id) in vehicles:
                detections_.append([x1,y1,x2,y2,score])
        
        #track vehicles
        track_ids=mot_tracker.update(np.asarray(detections_)) #it will match the new detections to the existing objects

        #detect license plates
        license_plates=license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id=license_plate

            #assign license plate to car 
            xcar1,ycar1,xcar2,ycar2,car_id=get_car(license_plate,track_ids)

            #crop license plate
            license_plate_crop=frame[int(y1):int(y2),int(x1):int(x2),:] #OpenCv Slicing under this format : image[y_start:y_end, x_start:x_end, channels]

            #process license plate
            license_plate_crop_gray=cv2.cvtColor(license_plate_crop,cv2.COLOR_BGR2GRAY) #Convert BGR color to GrayScale
            _,license_plate_crop_thresh=cv2.threshold(license_plate_crop_gray,64,255,cv2.THRESH_BINARY_INV) #Convert to black and white

            #read license plate number
            license_plate_text,license_plate_text_score=read_license_plate(license_plate_crop_thresh)
            
            if license_plate_text is not None:
                results[frame_nmr][car_id]={'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                            'license_plate': {'bbox': [x1, y1, x2, y2],
                                            'text': license_plate_text,
                                            'bbox_score': score,
                                            'text_score': license_plate_text_score}}
                
            # write results
            write_csv(results, r'E:\AutoParkAI\plateDetection\raw_data.csv')
