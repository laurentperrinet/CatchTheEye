#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Using psychopy to perform an experiment on the role of a bias in the direction """

import os
from LeCheapEyeTracker.EyeTrackerServer import Server
import imageio

et = Server()
camera = et.cam
import time

for i in range(5):
    print('i=', i)
    frame = camera.grab()
    imageio.imwrite(f'/tmp/{i}.png', frame[:, :, ::-1]) # converting on the fly from BGR > RGB
    time.sleep(1)
