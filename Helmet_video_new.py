# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:50:30 2021

@author: Ruturaj Nanoti
"""

from __future__ import print_function
import cv2 as cv
import argparse
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image

model = load_model('Helmet_Detection_md.h5')


def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    frame1 = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame1 = Image.fromarray(frame1)
    imt = frame1.resize((150, 150))
    imt = image.img_to_array(imt)
    imt = np.expand_dims(imt, axis=0)
    imt /= 255.
    k = model.predict_classes(imt)
    if (k[0][0] == 1):
        label = "No Helmet"
        color = (0, 0, 255)
    else:
        label = "Helmet is On"
        color = (0, 255, 0)
    for (x, y, w, h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv.imshow('Capture - Face detection', frame)


parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.',
                    default='/Users/Ruturaj Nanoti/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
face_cascade = cv.CascadeClassifier()

# -- 1. Load the cascades
if not face_cascade.load('/Users/Ruturaj Nanoti/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml'):
    print('--(!)Error loading face cascade')
    exit(0)

camera_device = args.camera

# -- 2. Read the video stream
cap = cv.VideoCapture("‘Smart Helmet’ for mass thermal screening of 200 people per minute BMC Smart Helmet.mp4")
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

#run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(1) == 3:
        break
