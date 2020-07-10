#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Using psychopy to perform an experiment on the role of a bias in the direction """

import os

import imageio

if True:
    from LeCheapEyeTracker.EyeTrackerServer import Server
    et = Server()
    camera = et.cam
else:
    import cv2
    camera = cv2.VideoCapture(0)
    def grab():
        ret, frame_bgr = camera.read()
        return ret, frame_bgr

import time

for i in range(5):
    print('i=', i+1)
    # frame = False
    frame = camera.grab()
    # while not (frame):
    #     ret, frame = camera.grab()
    #     print(ret, frame)
    imageio.imwrite(f'/tmp/{i}.png', frame[:, :, ::-1]) # converting on the fly from BGR > RGB
    time.sleep(1)
