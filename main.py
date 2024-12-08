# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
plt.ion()

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# Establish a connection to the webcam
# Number 0 is Iphone camera
# Number 1 is Mac webcam
cap = cv2.VideoCapture(1)
while cap.isOpened():
    # ret returns true or false depending on if frame was read successfully
    # frame captures the frame from the video capture device, it's a numpy array which represents image data
    ret, frame = cap.read()
    
    # Limits capture to 250 pixels and centers it
    frame = frame[415:415+250, 835:835+250, :]
    
    # Displays the captured image data onto the screen
    cv2.imshow("Image Feed", frame)
    
    # Breaking gracefully
    # It checks if the ascii value 'q' is pressed and if so breaks the capture loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()