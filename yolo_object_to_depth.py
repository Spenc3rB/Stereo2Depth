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

    
    # Resize the frames to match the input size expected by the model
    frame_right = cv2.resize(frame_right, (320, 320))
    frame_left = cv2.resize(frame_left, (320, 320))
    ########################################################################################

    # Convert the frame_right to float32 and normalize the values
    frame_right = frame_right.astype(np.float32)
    frame_right = frame_right / 255.0

    # Add a dimension to the frame_right so that it represents a batch
    frame_right = np.expand_dims(frame_right, axis=0)

    # If cannot catch any frame, break
    if succes_left and succes_right:                    

        # run YOLOv8 inference on both frames
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # [{'name': 'PartitionedCall:0', 'index': 45, 'shape': array([   1,   12, 2100], dtype=int32), 'shape_signature': array([   1,   12, 2100], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
        # [{'name': 'serving_default_images:0', 'index': 0, 'shape': array([  1, 320, 320,   3], dtype=int32), 'shape_signature': array([  1, 320, 320,   3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
        # print the details of the input and output tensors
        print("Input and output tensor shapes: ")
        print(input_details[0]['shape'])
        print(frame_right.shape)
        print(output_details[0]['shape'])
        # print the output tensor data type
        # [   1   12 2100] --> Model has 12 classes for each of the 2100 grid cells in the image
        break
