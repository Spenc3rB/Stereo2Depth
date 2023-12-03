import os
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# time for fps
import time

# import YOLO
from ultralytics import YOLO

# Open both cameras
cap_right = cv2.VideoCapture(2)                    
cap_left =  cv2.VideoCapture(0)

# Set camera resolution
cap_right.set(3, 320)
cap_right.set(4, 320)
cap_left.set(3, 320)
cap_left.set(4, 320)

# Stereo vision setup parameters
frame_rate = 30    #Camera frame rate (maximum at 120 fps)
B = 15               #Distance between the cameras [cm]
f = 3.67              #Camera lense's focal length [mm]
alpha = 70.42        #Camera field of view in the horizontal plane [degrees]

# YOLO setup parameters
MODEL = input("Enter model name: ")
model = YOLO(f"models/{MODEL}.pt") # results need stream=True

# Main program loop with face detector and depth estimation using stereo vision
while(cap_right.isOpened() and cap_left.isOpened()):

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

    ################## CALIBRATION #########################################################
    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

    ########################################################################################

    # If cannot catch any frame, break
    if succes_left and succes_right:                    

        # run YOLOv8 inference on both frames
        results = model(frame_right, imgsz=320)

        # get bounding boxes
        boxes = results.xyxy[0].numpy()

        # get center of bounding boxes
        centers = []
        for box in boxes:
            center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
            centers.append(center)

        # get depth of bounding boxes
        depths = []
        for center in centers:
            depth = tri.find_depth(center[0], center[1], frame_right, frame_left, B, f, alpha)
            depths.append(depth)
        
        # draw bounding boxes with depth on frame
        for i in range(len(boxes)):
            box = boxes[i]
            depth = depths[i]
            cv2.rectangle(frame_right, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(frame_right, f"{depth} cm", (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # show frame
        cv2.imshow("frame", frame_right)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break