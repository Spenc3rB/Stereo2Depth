import os
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import tflite_runtime.interpreter as tflite

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# time for fps
import time

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
MODEL = 'best_integer_quant_edgetpu.tflite'
interpreter = tflite.Interpreter(model_path=MODEL, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

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
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], frame_right)
        interpreter.invoke()
        results = interpreter.get_tensor(output_details[0]['index'])

        # print the output shape
        print(results.shape)
