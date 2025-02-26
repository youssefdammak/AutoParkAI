from ultralytics import YOLO

#load a model
model=YOLO("yolov8n.pt")

#use the model
results=model.train(data="E:/AutoParkAI/plateDetectionTraining/config.yaml",epochs=10) #train the model