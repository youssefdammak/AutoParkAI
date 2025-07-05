from ultralytics import YOLO
import cv2
from sort.sort import*
from util import*
from collections import Counter
import pandas as pd
import mysql.connector
from config import db_config
from flask import Flask, jsonify, make_response
from threading import Thread
from collections import Counter
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)
latest_plate = None  # Global variable to store latest detected plate
latest_entry_time=None
@app.route('/latest-plate', methods=['GET'])
def get_latest_plate():
    response = make_response(jsonify({"plate_number": latest_plate or None,"entry_time": latest_entry_time or None, "status":"Exit"}))
    response.headers["Cache-Control"] = "no-cache"
    return response

# Connect to MySQL
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

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

# Storage for detected plates
plate_detections = []

def run_video_processing():
    global latest_plate
    global latest_entry_time

    #read frames
    frame_nmr=-1
    ret = True

    while ret:
        frame_nmr+=1
        #ret : true if frame is captured
        #frame : numpy array for video frame(image)
        ret,frame=cap.read()
        
        if not ret:
            break

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

                #filter low confidence plates
                if score<0.5:
                    continue

                #assign license plate to car 
                xcar1,ycar1,xcar2,ycar2,car_id=get_car(license_plate,track_ids)

                #crop license plate
                license_plate_crop=frame[int(y1):int(y2),int(x1):int(x2),:] #OpenCv Slicing under this format : image[y_start:y_end, x_start:x_end, channels]

                #process license plate
                license_plate_crop_gray=cv2.cvtColor(license_plate_crop,cv2.COLOR_BGR2GRAY) #Convert BGR color to GrayScale
                _,license_plate_crop_thresh=cv2.threshold(license_plate_crop_gray,64,255,cv2.THRESH_BINARY_INV) #Convert to black and white

                #read license plate number
                license_plate_text,license_plate_text_score=read_license_plate(license_plate_crop_thresh)
                
                if license_plate_text:
                    plate_detections.append((license_plate_text,score))
                
                if frame_nmr%10==0 and license_plate_text:
                    # Count occurrences of each plate
                    plate_counts = Counter(plate for plate, _ in plate_detections)

                    # Get the most frequent plate
                    most_frequent_plate,occurance_count = plate_counts.most_common(1)[0]

                    # Get the highest confidence score for that plate
                    highest_score = max(score for plate, score in plate_detections if plate == most_frequent_plate)

                    # Check if the plate exists
                    sql_check = "SELECT * FROM ParkingActivity WHERE plate_number = %s ORDER BY entry_time DESC LIMIT 1"
                    cursor.execute(sql_check, (most_frequent_plate,))
                    row=cursor.fetchone()
                    
                    if highest_score>0.5 and occurance_count>4 and row is not None and row[4] is None:

                        #Find user ID from plate number
                        sql_get_user = "SELECT id FROM Users WHERE plate_number = %s"
                        cursor.execute(sql_get_user, (most_frequent_plate,))
                        user_row = cursor.fetchone()

                        user_id = user_row[0] if user_row else None

                        exit_time = datetime.now()
                        entry_time = row[3]

                        # Calculate duration in hours
                        duration_hours = (exit_time - entry_time).total_seconds() / 3600

                        # Set a rate (e.g., $2/hour)
                        rate = 2.00
                        amount = round(duration_hours * rate, 2)

                        sql_insert = """
                            UPDATE ParkingActivity
                            SET exit_time = %s,
                                amount = %s
                            WHERE plate_number = %s AND entry_time = %s;
                        """
                        exit_time=datetime.now()
                        cursor.execute(sql_insert, (exit_time, amount, most_frequent_plate, entry_time))
                        conn.commit()

                        sql_insert = "INSERT INTO Payments (user_id, amount) VALUES (%s, %s)"
                        cursor.execute(sql_insert, (user_id, amount))
                        conn.commit()

                        latest_plate = most_frequent_plate
                        latest_entry_time=exit_time

                    plate_detections.clear()

                if license_plate_text is not None:
                    results[frame_nmr][car_id]={'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                'text': license_plate_text,
                                                'bbox_score': score,
                                                'text_score': license_plate_text_score}}
                    
                # write results
                write_csv(results, r'E:\AutoParkAI\plateDetection\raw_data.csv')

def run_flask():
    app.run(host='0.0.0.0', port=5001)

flask_thread = Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

run_video_processing()