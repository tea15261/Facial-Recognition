# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# Import uuid library to generate unique image names
#uuid (universally unique identifier) 
import uuid

# Set up paths
ANC_PATH = os.path.join('data', 'anchor')
POS_PATH = os.path.join('data', 'positive')

# Establish a connection to the webcam
# Number 0 is Iphone camera
# Number 1 is Mac webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    # ret returns true or false depending on if frame was read successfully
    # frame captures the frame from the video capture device, it's a numpy array which represents image data
    ret, frame = cap.read()
    
    # Limits capture to 250 pixels and centers it
    frame = frame[415:415+250, 835:835+250, :]
    
    # Displays the captured image data onto the screen
    cv2.imshow("Image Feed", frame)
    
    # Capture anchors
    if cv2.waitKey(1) & 0xFF == ord('a'):
        # create the unique file path
        img_name = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # write out anchor image
        cv2.imwrite(img_name, frame)
    
    # Capture positives
    if cv2.waitKey(1) & 0xFF == ord('p'):
        # create the unique file path
        img_name = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # write out positive image
        cv2.imwrite(img_name, frame)
    
    # Breaking gracefully
    # It checks if the ascii value 'q' is pressed and if so breaks the capture loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()